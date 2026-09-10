"""Re-run the deterministic conversation with a single decision intervention."""
from contextlib import contextmanager
from .engine import conversation
from .interventions import replacement


def evaluate(record,target,before,scenario,texts,runtimes,task,layer=-1):
    runtime=runtimes[target['model']];blocks=runtime.model.model.layers;layer=layer%len(blocks)
    probe=texts['stance_probe'].format(candidate_1=scenario['strategy_1']['candidate'],candidate_2=scenario['strategy_2']['candidate'])
    context=before+[{'role':'user','content':probe}]
    pos=len(runtime.tokenizer(runtime.inputs(context)[0])['input_ids'])-1
    captured=[]
    def capture(m,args,output):
        tensor=output[0] if isinstance(output,tuple) else output
        captured.append(tensor[0,pos].detach().clone())
    handle=blocks[layer].register_forward_hook(capture)
    try:runtime.likelihood(context,scenario['strategy_1']['candidate'])
    finally:handle.remove()
    applied=[]
    @contextmanager
    def intervention(t,agent,model,history):
        if t==target['round'] and agent==target['agent']:
            if history!=target['decision_history']:raise ValueError('Continuation did not reproduce target history')
            prompt=history+[{'role':'user','content':probe}]
            recipient=len(model.tokenizer(model.inputs(prompt)[0])['input_ids'])-1
            with replacement(blocks[layer],recipient,captured[0]):
                applied.append(True);yield
        else:yield
    rows=conversation(record['config'],scenario,texts,runtimes,decision_context=intervention)
    if len(applied)!=1:raise ValueError('Expected exactly one intervention')
    # Establish unchanged decisions before intervention; do not claim GPU equivalence by assumption.
    index=next(i for i,r in enumerate(record['rows']) if r['round']==target['round'] and r['agent']==target['agent'])
    if any(a['message']!=b['message'] for a,b in zip(record['rows'][:index],rows[:index])):raise ValueError('Baseline replay changed before intervention')
    return dict(task=task,agent=target['agent'],round=target['round'],layer=layer,donor='before_peer',recipient='decision',
                baseline=[dict(round=r['round'],agent=r['agent'],strategy=r['strategy'],margin=r['post']['margin']) for r in record['rows']],rows=rows)
