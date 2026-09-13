"""Standalone scored-choice initialization pilot; no conversation continuation."""
import argparse, csv, hashlib, json, sys
from contextlib import contextmanager
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from llm_mas.runtime import Runtime
from llm_mas.prompts import initial, for_condition

OUT = Path(__file__).resolve().parent

def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2))

def table(name, rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

@contextmanager
def observe(blocks, position, vectors):
    handles=[]
    def hook(layer):
        def capture(module,args,output):
            tensor=output[0] if isinstance(output,tuple) else output
            vectors[layer]=tensor[0,position].detach().cpu().clone()
        return capture
    try:
        for i,b in enumerate(blocks):handles.append(b.register_forward_hook(hook(i)))
        yield
    finally:
        for h in handles:h.remove()

@contextmanager
def patch(block,position,vector):
    # One fresh context per likelihood call. Scoring is teacher forced, with
    # use_cache=False in Runtime. Causality prevents candidate suffix leakage.
    called=[False]
    def hook(module,args,output):
        if called[0]:return output
        tensor=output[0] if isinstance(output,tuple) else output
        if position>=tensor.shape[1]:raise ValueError('Missing prompt position')
        called[0]=True
        changed=tensor.clone();changed[:,position]=vector.to(tensor)
        return (changed,)+output[1:] if isinstance(output,tuple) else changed
    handle=block.register_forward_hook(hook)
    try:yield
    finally:handle.remove()

def plot_results(cfg, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,case in enumerate(cfg['cases']):
        for direction in ('forward_to_reversed','reversed_to_forward'):
            rows=[r for r in results if r['case']==case['name'] and r['direction']==direction and r['control']=='donor']
            axes[i,0].plot([r['layer'] for r in rows],[r['strategy'] for r in rows],'.-',label=direction)
            axes[i,1].plot([r['layer'] for r in rows],[r['margin_delta'] for r in rows],'.-',label=direction)
        axes[i,0].set(title=case['name']+' — patched strategy',yticks=[1,2],ylabel='Semantic strategy')
        axes[i,1].set(title=case['name']+' — margin change',ylabel='Patched − recipient margin');axes[i,1].axhline(0,color='gray',lw=.6)
        for ax in axes[i]:ax.set_xlabel('Transformer block (zero indexed)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'layer_transfer.png',dpi=180);fig.savefig(OUT/'layer_transfer.pdf');plt.close(fig)

def pilot_device(torch, requested):
    """Prefer CUDA, then MPS; never silently select CPU."""
    if requested == 'cpu':
        return 'cpu'
    available = dict(cuda=torch.cuda.is_available(),
                     mps=hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    if requested == 'auto':
        for device in ('cuda', 'mps'):
            if available[device]:
                return device
    elif available.get(requested, False):
        return requested
    raise RuntimeError(f'Requested GPU device {requested} is unavailable. '
                       'Use a CUDA or MPS-enabled PyTorch environment. '
                       'CPU diagnostics require explicit --device cpu; '
                       'automatic CPU fallback is disabled.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=('cuda','mps','auto','cpu'),default='auto',help='Auto prefers CUDA then MPS; never falls back to CPU. CPU is explicit diagnostics only.');ap.add_argument('--tolerance',type=float,default=1e-4);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--plot-only',action='store_true');args=ap.parse_args()
    cfg=json.loads((OUT/'config.json').read_text())
    if args.plot_only:
        rows=list(csv.DictReader((OUT/'patches.csv').open()))
        for row in rows:
            for key in ('layer','strategy'):row[key]=int(row[key])
            row['margin_delta']=float(row['margin_delta'])
        plot_results(cfg,rows)
        return
    inputs=[];baseline=[]
    for case in cfg['cases']:
        for direction in ('forward','reversed'):
            path=ROOT/case[direction];d=json.loads(path.read_text());c=d['config'];scenario=d['scenario']
            assert c['initialization']=='natural' and c['decision_mode']=='scored'
            row=next(r for r in d['rows'] if r['round']==0 and r['agent']==case['agent'])
            texts=json.loads((path.parent.parent/'inputs/prompts.json').read_text())
            texts=for_condition(texts,c,scenario)
            idx=ord(case['agent'][-1])-65
            history=initial(scenario,texts,c['presentation_order'],c['agents'],persona=idx%2+1 if c.get('persona') else None)
            assert history==row['decision_history'], 'Initialization reconstruction mismatch'
            probe=texts['stance_probe'].format(candidate_1=scenario['strategy_1']['candidate'],candidate_2=scenario['strategy_2']['candidate'])
            inputs.append(dict(case=case['name'],direction=direction,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),config=c,scenario=scenario,row=row,texts=texts,history=history,messages=history+[dict(role='user',content=probe)]))
            baseline.append(dict(case=case['name'],direction=direction,agent=case['agent'],model=row['model'],scenario=scenario['id'],persona=c.get('persona',False),initialization=c['initialization'],procedure=c['decision_mode'],schedule=c['schedule'],strategy=row['strategy'],likelihood_1=row['decision']['likelihoods'][0],likelihood_2=row['decision']['likelihoods'][1],margin=row['decision']['margin'],message=row['message'],source=str(path.relative_to(ROOT))))
    # Require identical conditions except the two presentation manipulations.
    for f,r in zip(inputs[::2],inputs[1::2]):
        assert f['scenario']==r['scenario']
        for key in ('models','persona','initialization','decision_mode','schedule','argument_mode','max_argument_tokens','decoding'):
            assert f['config'].get(key)==r['config'].get(key),key
        assert not f['config'].get('reverse_candidate_blocks',False) and r['config']['reverse_candidate_blocks']
        assert f['config']['presentation_order']=='strategy_1_first' and r['config']['presentation_order']=='strategy_2_first'
    table('matched_cases.csv',baseline);save('inputs.json',inputs)
    if args.prepare_only:return
    import torch, transformers
    device=pilot_device(torch,args.device)
    gpu_name=torch.cuda.get_device_name() if device=='cuda' else ('Apple GPU (MPS)' if device=='mps' else None)
    print(f'Pilot device: {device}' + (f' — {gpu_name}' if gpu_name else ' (explicit CPU diagnostics)'),flush=True)
    runtime=Runtime(baseline[0]['model'],device)
    if any(p.device.type != device for p in runtime.model.parameters()):
        raise RuntimeError('Model parameters are not all on the requested device')
    save('environment.json',dict(torch=torch.__version__,transformers=transformers.__version__,device=runtime.device,gpu_name=gpu_name,cuda_version=torch.version.cuda,gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if device=='cuda' else None,dtype=str(next(runtime.model.parameters()).dtype),model_revision=getattr(runtime.model.config,'_commit_hash',None),tolerance=args.tolerance))
    replay=[]
    for item in inputs:
        m=runtime.measure(item['history'],item['scenario'],item['texts'])
        selected=1 if m['margin']>=0 else 2
        prefix=item['scenario'][f'strategy_{selected}']['candidate']+'\n'
        gen=runtime.generate(item['messages'],item['config']['max_argument_tokens'],continuation_prefix=prefix)
        message=prefix+gen['output']
        post=runtime.measure(item['history']+[dict(role='assistant',content=message)],item['scenario'],item['texts'])
        saved=item['row']
        errors=[abs(a-b) for a,b in zip(m['likelihoods'],saved['decision']['likelihoods'])]
        post_error=max(abs(a-b) for a,b in zip(post['likelihoods'],saved['post']['likelihoods']))
        rendered_equal=m['rendered_prompt']==saved['decision']['rendered_prompt'] and gen['rendered_prompt']==saved['generation']['rendered_prompt']
        row=dict(case=item['case'],direction=item['direction'],prompt_exact=rendered_equal,response_exact=message==saved['message'],strategy=selected,score_error_max=max(errors),post_score_error_max=post_error,likelihood_1=m['likelihoods'][0],likelihood_2=m['likelihoods'][1],margin=m['margin'],response=message)
        row['pass']=rendered_equal and row['response_exact'] and max(errors)<=args.tolerance and post_error<=args.tolerance
        replay.append(row);table('replay.csv',replay)
        print('Replay',item['case'],item['direction'],row['pass'],flush=True)
    if not all(r['pass'] for r in replay):
        save('status.json',dict(status='replay_mismatch',message='No activation replacements run. Inspect replay.csv; exact responses and scores within tolerance required.'))
        return
    blocks=runtime.model.model.layers;captures=[];controls=[]
    for item,base in zip(inputs,replay):
        _,encoded=runtime.inputs(item['messages']);position=encoded.input_ids.shape[1]-1
        vectors={}
        with observe(blocks,position,vectors):
            first=runtime.likelihood(item['messages'],item['scenario']['strategy_1']['candidate'])[0]
        # Second baseline call independently tests observation of the other score.
        other={}
        with observe(blocks,position,other):
            second=runtime.likelihood(item['messages'],item['scenario']['strategy_2']['candidate'])[0]
        err=max(abs(first-base['likelihood_1']),abs(second-base['likelihood_2']))
        causal_error=max((vectors[i]-other[i]).abs().max().item() for i in vectors)
        controls.append(dict(case=item['case'],direction=item['direction'],score_error=err,candidate_suffix_activation_error=causal_error,pass_check=err<=args.tolerance and causal_error<=args.tolerance))
        table('observation_controls.csv',controls)
        if not controls[-1]['pass_check']:raise ValueError('Observation control failed')
        torch.save(dict(position=position,hidden_states=vectors),OUT/f"activations_{item['case']}_{item['direction']}.pt")
        captures.append((position,vectors))
    results=[]
    for start in range(0,len(inputs),2):
        for recipient,donor in ((start+1,start),(start,start+1)):
            item=inputs[recipient];base=replay[recipient];target=replay[donor]
            position,_=captures[recipient]
            for layer,block in enumerate(blocks):
                for control,source in (('self',recipient),('donor',donor)):
                    scores=[]
                    for k in (1,2):
                        with patch(block,position,captures[source][1][layer]):
                            scores.append(runtime.likelihood(item['messages'],item['scenario'][f'strategy_{k}']['candidate'])[0])
                    margin=scores[0]-scores[1];strategy=1 if margin>=0 else 2
                    error=max(abs(scores[0]-base['likelihood_1']),abs(scores[1]-base['likelihood_2']))
                    results.append(dict(case=item['case'],direction=inputs[donor]['direction']+'_to_'+item['direction'],layer=layer,control=control,likelihood_1=scores[0],likelihood_2=scores[1],margin=margin,margin_delta=margin-base['margin'],strategy=strategy,recipient_strategy=base['strategy'],donor_strategy=target['strategy'],matches_donor=int(strategy==target['strategy']),changed_strategy=int(strategy!=base['strategy']),self_score_error=error if control=='self' else '',position=position))
                    table('patches.csv',results)
                    if control=='self' and (error>args.tolerance or strategy!=base['strategy']):raise ValueError('Self replacement failed; stop interpretation')
                print(item['case'],results[-1]['direction'],'layer',layer,flush=True)
    plot_results(cfg, results)
    hits=[r for r in results if r['control']=='donor' and r['recipient_strategy']!=r['donor_strategy'] and r['matches_donor']]
    text='Scored-choice intervention only. Both full endorsement likelihoods are measured independently without shared caches.\n\n'
    text+=('Transfer layers: '+str([(r['direction'],r['layer']) for r in hits])+'. These blocks warrant finer attention/MLP tests; this pilot does not localize a unique mechanism.' if hits else 'No block transferred the changed choice. Margin effects may motivate finer tests, but this pilot provides no strategy-transfer layer range.')
    text+='\n\nThe unchanged case is a specificity control: matching its donor is not evidence of transfer. Findings concern selected initialization prompts, not later interaction or population frequencies.\n'
    (OUT/'interpretation.md').write_text(text);save('status.json',dict(status='complete',patch_rows=len(results)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        save('status.json',dict(status='failed',error=str(error)))
        raise
