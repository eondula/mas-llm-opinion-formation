"""Confirm heads 2/9 using repeated dose grids and token-level endorsement scores."""
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

# Reuse tested hook lifecycle and GPU selection from the first standalone pilot.
import importlib.util
_spec=importlib.util.spec_from_file_location('reversal_pilot',ROOT/'results/presentation_reversal_patching/run.py')
_base=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
observe, patch, pilot_device = _base.observe, _base.patch, _base.pilot_device



# Reuse head capture and validated placement utilities.
_hspec=importlib.util.spec_from_file_location('head_pilot',ROOT/'results/presentation_reversal_heads/run.py')
_head=importlib.util.module_from_spec(_hspec);_hspec.loader.exec_module(_head)
capture_pre=_head.capture_pre

@contextmanager
def interpolate(module,position,donor,heads,alphas,head_dim):
    calls=[0]
    def hook(m,args):
        if calls[0]:return args
        calls[0]+=1;x=args[0];changed=x.clone()
        if donor.shape!=x[0,position].shape:raise ValueError('Invalid donor shape')
        for head,alpha in zip(heads,alphas):
            if not 0<=alpha<=1:raise ValueError('Invalid replacement strength')
            sl=slice(head*head_dim,(head+1)*head_dim)
            if alpha==0:continue
            if alpha==1:changed[:,position,sl]=donor[sl].to(x)
            else:
                # Interpolate in fp32, then return to the existing execution dtype.
                mixed=(1-alpha)*x[:,position,sl].float()+alpha*donor[sl].to(x.device).float()
                changed[:,position,sl]=mixed.to(x.dtype)
        return (changed,)+args[1:]
    handle=module.register_forward_pre_hook(hook)
    try:yield
    finally:handle.remove()
    if calls[0]!=1:raise ValueError('Interpolation hook did not execute')

def plot_results():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((OUT/'grid.csv').open()))
    tokens=list(csv.DictReader((OUT/'token_scores.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(11,9))
    for i,case in enumerate(('changed','unchanged')):
        for j,direction in enumerate(('forward_to_reversed','reversed_to_forward')):
            ax=axes[i,j];g=[r for r in rows if r['case']==case and r['direction']==direction and r['control']=='donor' and r['repeat']=='0']
            matrix=[[float(next(r['margin'] for r in g if float(r['alpha_2'])==a and float(r['alpha_9'])==b)) for b in (0,.5,1)] for a in (0,.5,1)]
            limit=max(abs(v) for row in matrix for v in row) or 1
            im=ax.imshow(matrix,origin='lower',cmap='coolwarm',vmin=-limit,vmax=limit)
            for y,row in enumerate(matrix):
                for x,v in enumerate(row):ax.text(x,y,f'{v:+.3f}',ha='center',va='center',color='black')
            ax.set(title=case+' / '+direction,xlabel='Head 9 strength',ylabel='Head 2 strength',xticks=[0,1,2],yticks=[0,1,2],xticklabels=[0,.5,1],yticklabels=[0,.5,1]);fig.colorbar(im,ax=ax,label='S1 minus S2 log-likelihood')
    fig.tight_layout();fig.savefig(OUT/'joint_margin_grid.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(15,15))
    for i,(case,direction) in enumerate(( (c,d) for c in ('changed','unchanged') for d in ('forward_to_reversed','reversed_to_forward'))):
        for k in (1,2):
            ax=axes[i,k-1]
            for a,b,label in [(1,0,'head 2'),(0,1,'head 9'),(1,1,'joint')]:
                g=[r for r in tokens if r['case']==case and r['direction']==direction and int(r['candidate'])==k and r['control']=='donor' and r['repeat']=='0' and float(r['alpha_2'])==a and float(r['alpha_9'])==b]
                ax.plot([int(r['token_index']) for r in g],[float(r['logprob_delta']) for r in g],'.-',label=label)
            ax.axhline(0,color='gray',lw=.5)
            if g:
                shared=sum(r['region']=='shared_prefix' for r in g)
                ax.axvline(shared-.5,color='gray',ls=':',label='first differing token')
                ax.set_xticks([int(r['token_index']) for r in g]);ax.set_xticklabels([r['token'] for r in g],rotation=65,ha='right',fontsize=7)
            ax.set(title=f'{case} / {direction} / S{k}',ylabel='Token log-likelihood change');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'token_effects.png',dpi=180);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=('cuda','mps','auto','cpu'),default='auto',help='Auto prefers CUDA then MPS; never falls back to CPU. CPU is explicit diagnostics only.');ap.add_argument('--tolerance',type=float,default=1e-4);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--plot-only',action='store_true');args=ap.parse_args()
    cfg=json.loads((OUT/'config.json').read_text())
    if cfg['procedure']!='scored' or cfg['layer']!=15 or cfg['heads']!=[2,9]:
        raise ValueError('This pilot requires scored block 15 and heads 2/9')
    if cfg['alphas']!=[0,0.5,1] or cfg['repeats']<2:
        raise ValueError('Expected a 3x3 grid with at least two repeats')
    if args.plot_only:
        if json.loads((OUT/'status.json').read_text())['status']!='complete':raise ValueError('Require completed run')
        plot_results()
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
    save('status.json',dict(status='running',phase='replay'))
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
    layer=cfg['layer'];projection=runtime.model.model.layers[layer].self_attn.o_proj
    nheads=runtime.model.config.num_attention_heads;dim=projection.in_features//nheads
    if projection.in_features!=nheads*dim or max(cfg['heads'])>=nheads:raise ValueError('Invalid head geometry')
    save('head_layout.json',dict(layer=layer,heads=nheads,head_dim=dim,selected=cfg['heads'],interpolation='fp32 convex combination cast back to recipient dtype; exact endpoint assignment'))
    references=list(csv.DictReader((ROOT/cfg['reference']).open()))
    save('reference_provenance.json',dict(path=cfg['reference'],sha256=hashlib.sha256((ROOT/cfg['reference']).read_bytes()).hexdigest()))
    captures=[];baseline_tokens=[];observations=[];token_baselines=[]
    for item,base in zip(inputs,replay):
        rendered,encoded=runtime.inputs(item['messages']);position=encoded.input_ids.shape[1]-1
        values=[];vectors=[];ids=[]
        for k in (1,2):
            candidate=item['scenario'][f'strategy_{k}']['candidate'];sink=[]
            with capture_pre(projection,position,sink):score,token_values,_=runtime.likelihood(item['messages'],candidate)
            vectors.append(sink[0]);values.append(token_values)
            _,full=runtime.inputs(item['messages'],candidate);ids.append(full.input_ids[0,position+1:].tolist())
            if abs(score-base[f'likelihood_{k}'])>args.tolerance:raise ValueError('Observation changed score')
        suffix_error=(vectors[0]-vectors[1]).abs().max().item()
        observations.append(dict(case=item['case'],direction=item['direction'],suffix_error=suffix_error,pass_check=suffix_error<=args.tolerance))
        table('observation_controls.csv',observations)
        if suffix_error>args.tolerance:raise ValueError('Prompt capture depends on candidate suffix')
        common=0
        for a,b in zip(*ids):
            if a!=b:break
            common+=1
        for k in (1,2):
            for t,(tid,val) in enumerate(zip(ids[k-1],values[k-1])):
                token_baselines.append(dict(case=item['case'],presentation=item['direction'],candidate=k,token_index=t,token_id=tid,token=runtime.tokenizer.convert_ids_to_tokens(tid),region='shared_prefix' if t<common else 'remaining_suffix',logprob=val))
        captures.append((position,vectors[0],ids,common));baseline_tokens.append(values)
        torch.save(dict(position=position,pre_projection=vectors[0],head_dim=dim),OUT/f"activations_{item['case']}_{item['direction']}.pt")
    table('baseline_tokens.csv',token_baselines)
    results=[];tokens=[];confirm=[];first={}
    corners={(0,0):('all','self'),(1,0):('single','2'),(0,1):('single','9'),(1,1):('joint','2,9')}
    for recipient,donor in [(1,0),(0,1),(3,2),(2,3)]:
        item=inputs[recipient];base=replay[recipient];direction=inputs[donor]['direction']+'_to_'+item['direction']
        for repeat in range(cfg['repeats']):
            for a in cfg['alphas']:
                for b in cfg['alphas']:
                    for control in ('self','donor'):
                        source=recipient if control=='self' else donor;scores=[];tv=[]
                        for k in (1,2):
                            with interpolate(projection,captures[recipient][0],captures[source][1],cfg['heads'],[a,b],dim):
                                score,vals,_=runtime.likelihood(item['messages'],item['scenario'][f'strategy_{k}']['candidate'])
                            scores.append(score);tv.append(vals)
                        error=max(abs(scores[k-1]-base[f'likelihood_{k}']) for k in (1,2))
                        signature=(item['case'],direction,a,b,control)
                        reference=first.setdefault(signature,(scores,tv))
                        repeat_error=max(abs(x-y) for v,w in zip(tv,reference[1]) for x,y in zip(v,w))
                        margin=scores[0]-scores[1]
                        row=dict(case=item['case'],direction=direction,repeat=repeat,alpha_2=a,alpha_9=b,control=control,likelihood_1=scores[0],likelihood_2=scores[1],margin=margin,margin_delta=margin-base['margin'],strategy=1 if margin>=0 else 2,recipient_margin=base['margin'],repeat_token_error=repeat_error,self_score_error=error if control=='self' or (a,b)==(0,0) else '')
                        results.append(row);table('grid.csv',results)
                        for k in (1,2):
                            ids=captures[recipient][2][k-1]
                            if len(tv[k-1])!=len(ids):raise ValueError('Token alignment changed')
                            for t,(tid,val,bval) in enumerate(zip(ids,tv[k-1],baseline_tokens[recipient][k-1])):
                                tokens.append(dict(case=item['case'],direction=direction,repeat=repeat,alpha_2=a,alpha_9=b,control=control,candidate=k,token_index=t,token_id=tid,token=runtime.tokenizer.convert_ids_to_tokens(tid),region='shared_prefix' if t<captures[recipient][3] else 'remaining_suffix',logprob=val,baseline_logprob=bval,logprob_delta=val-bval,signed_margin_delta=(val-bval)*(1 if k==1 else -1)))
                        table('token_scores.csv',tokens)
                        if (control=='self' or (a,b)==(0,0)) and error>args.tolerance:raise ValueError('Self/zero control failed')
                        if repeat_error>args.tolerance:raise ValueError('Repeated token scores differ')
                        if control=='donor' and (a,b) in corners:
                            kind,label=corners[a,b]
                            refs=[r for r in references if r['case']==item['case'] and r['direction']==direction and r['kind']==kind and (r['control']=='self' if label=='self' else r['control']=='donor' and r['heads']==label)]
                            if len(refs)!=1:raise ValueError('Missing unique prior reference')
                            e=max(abs(scores[k-1]-float(refs[0][f'likelihood_{k}'])) for k in (1,2))
                            confirm.append(dict(case=item['case'],direction=direction,repeat=repeat,alpha_2=a,alpha_9=b,prior_score_error=e,pass_check=e<=args.tolerance));table('confirmation_controls.csv',confirm)
                            if e>args.tolerance:raise ValueError('Prior endpoint did not reproduce')
            print(item['case'],direction,'repeat',repeat+1,flush=True)
    # Contributions retain token identity; suffix includes every token after divergence,
    # not just attribute words. Compare candidates via signed score contributions.
    contributions=[]
    for row in results:
        if row['control']!='donor':continue
        relevant=[t for t in tokens if all(t[k]==row[k] for k in ('case','direction','repeat','alpha_2','alpha_9','control'))]
        prefix=sum(t['signed_margin_delta'] for t in relevant if t['region']=='shared_prefix')
        suffix=sum(t['signed_margin_delta'] for t in relevant if t['region']=='remaining_suffix')
        if abs(prefix+suffix-row['margin_delta'])>args.tolerance:raise ValueError('Token deltas do not sum to margin change')
        contributions.append(dict(case=row['case'],direction=row['direction'],repeat=row['repeat'],alpha_2=row['alpha_2'],alpha_9=row['alpha_9'],shared_prefix_margin_delta=prefix,remaining_suffix_margin_delta=suffix,total_margin_delta=prefix+suffix))
    table('token_contributions.csv',contributions)
    plot_results()
    (OUT/'interpretation.md').write_text('All baseline replay, capture, self/zero, repeat and prior-endpoint checks passed. Inspect grid.csv and token_contributions.csv for the measured dose response and token attribution. Token scores show where the output likelihood changes, not a linguistic role or internal causal mediation. Three identical greedy repeats are numerical consistency checks, not independent experimental replications. Figures show repeat 0 after verifying agreement. No additive prediction replaces measured joint scores.\n')
    save('status.json',dict(status='complete',grid_rows=len(results),token_rows=len(tokens)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        save('status.json',dict(status='failed',error=str(error)))
        raise
