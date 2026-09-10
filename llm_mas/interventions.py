"""Within-checkpoint patches at the final prompt position, with explicit controls."""
from contextlib import contextmanager

@contextmanager
def replacement(module, position, vector):
    def hook(_module, inputs, output):
        tensor=output[0] if isinstance(output,tuple) else output
        if not 0<=position<tensor.shape[1]:raise ValueError('Patch position outside sequence')
        if tensor[0,position].shape!=vector.shape:raise ValueError('Incompatible donor activation')
        changed=tensor.clone();changed[:,position]=vector.to(tensor)
        return (changed,)+output[1:] if isinstance(output,tuple) else changed
    handle=module.register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def sweep(runtime, before, after, scenario, texts, layers=None):
    blocks=getattr(getattr(runtime.model,'model',None),'layers',None)
    if blocks is None:raise ValueError('Mechanistic adapter requires model.model.layers')
    selected=layers if layers is not None else range(len(blocks))
    candidates=[scenario[k]['candidate'] for k in ('strategy_1','strategy_2')]
    probe=texts['stance_probe'].format(candidate_1=candidates[0],candidate_2=candidates[1])
    contexts=[h+[{'role':'user','content':probe}] for h in (before,after)]
    positions=[len(runtime.tokenizer(runtime.inputs(h)[0])['input_ids'])-1 for h in contexts]
    base=[runtime.measure(h,scenario,texts)['margin'] for h in (before,after)]
    records=[]
    for layer in selected:
        block=blocks[layer]
        for component,module in [('residual',block),('attention',getattr(block,'self_attn',None)),('mlp',getattr(block,'mlp',None))]:
            if module is None:continue
            vectors=[]
            for h,pos in zip(contexts,positions):
                captured=[]
                def capture(_module,inputs,output):
                    tensor=output[0] if isinstance(output,tuple) else output
                    captured.append(tensor[0,pos].detach().clone())
                handle=module.register_forward_hook(capture)
                try:runtime.likelihood(h,candidates[0])
                finally:handle.remove()
                vectors.append(captured[0])
            for recipient,donor in [(0,1),(1,0)]:
                for control,vector in [('donor',vectors[donor]),('self',vectors[recipient]),('zero',vectors[recipient]*0)]:
                    with replacement(module,positions[recipient],vector):
                        score=runtime.measure((before,after)[recipient],scenario,texts)['margin']
                    denominator=base[donor]-base[recipient]
                    records.append(dict(layer=layer,component=component,recipient=recipient,donor=donor,control=control,
                        margin=score,baseline=base[recipient],donor_margin=base[donor],delta=score-base[recipient],
                        recovery=(score-base[recipient])/denominator if abs(denominator)>1e-8 else None,
                        position=positions[recipient],anchor='final_prompt_token'))
    return records

@contextmanager
def input_replacement(module, position, vector, head=None, head_dim=None):
    def hook(_module, inputs):
        changed=inputs[0].clone()
        sl=slice(None) if head is None else slice(head*head_dim,(head+1)*head_dim)
        changed[:,position,sl]=vector[sl].to(changed)
        return (changed,)+inputs[1:]
    handle=module.register_forward_pre_hook(hook)
    try:yield
    finally:handle.remove()


def head_sweep(runtime,before,after,scenario,texts,layers=None):
    blocks=runtime.model.model.layers
    probe=texts['stance_probe'].format(candidate_1=scenario['strategy_1']['candidate'],candidate_2=scenario['strategy_2']['candidate'])
    contexts=[h+[{'role':'user','content':probe}] for h in (before,after)]
    positions=[len(runtime.tokenizer(runtime.inputs(h)[0])['input_ids'])-1 for h in contexts]
    base=[runtime.measure(h,scenario,texts)['margin'] for h in (before,after)]
    effects=[]
    for layer in layers if layers is not None else range(len(blocks)):
        attention=blocks[layer].self_attn; projection=attention.o_proj
        heads=runtime.model.config.num_attention_heads
        vectors=[]
        for h,pos in zip(contexts,positions):
            capture=[]
            handle=projection.register_forward_pre_hook(lambda m,args: capture.append(args[0][0,pos].detach().clone()))
            try:runtime.likelihood(h,scenario['strategy_1']['candidate'])
            finally:handle.remove()
            vectors.append(capture[0])
        width=vectors[0].shape[-1]
        if width%heads:raise ValueError('Attention output cannot be divided into heads')
        for head in range(heads):
            for recipient,donor in ((0,1),(1,0)):
                with input_replacement(projection,positions[recipient],vectors[donor],head,width//heads):
                    margin=runtime.measure((before,after)[recipient],scenario,texts)['margin']
                effects.append(dict(layer=layer,head=head,recipient=recipient,margin=margin,baseline=base[recipient],delta=margin-base[recipient]))
    return effects


def source_span_sweep(runtime,before,after,scenario,texts,layers=None):
    """Zero residual outputs over peer-message token spans; not a head-path claim."""
    probe=texts['stance_probe'].format(candidate_1=scenario['strategy_1']['candidate'],candidate_2=scenario['strategy_2']['candidate'])
    h=after+[{'role':'user','content':probe}]
    rendered=runtime.inputs(h)[0]
    encoded=runtime.tokenizer(rendered,return_offsets_mapping=True)
    offsets=encoded['offset_mapping'];spans=[]
    prefix=runtime.tokenizer.apply_chat_template(before,tokenize=False,add_generation_prompt=False)
    if not rendered.startswith(prefix):raise ValueError('Cannot align retained history for source spans')
    search_from=len(prefix)
    for message in after[len(before):]:
        start=rendered.find(message['content'],search_from)
        if start<0:raise ValueError('Peer message not found in rendered prompt')
        end=start+len(message['content']);search_from=end
        indices=[i for i,(a,b) in enumerate(offsets) if b>start and a<end and b>a]
        spans.append(indices)
    baseline=runtime.measure(after,scenario,texts)['margin'];effects=[]
    blocks=runtime.model.model.layers
    for layer in layers if layers is not None else range(len(blocks)):
        for span,indices in enumerate(spans):
            def hook(module,inputs,output):
                tensor=output[0] if isinstance(output,tuple) else output
                changed=tensor.clone();changed[:,indices]=0
                return (changed,)+output[1:] if isinstance(output,tuple) else changed
            handle=blocks[layer].register_forward_hook(hook)
            try:margin=runtime.measure(after,scenario,texts)['margin']
            finally:handle.remove()
            effects.append(dict(layer=layer,span=span,token_indices=indices,baseline=baseline,margin=margin,delta=margin-baseline,intervention='zero_peer_span_residual'))
    return effects
