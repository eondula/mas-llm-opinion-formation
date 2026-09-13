"""Block-15 pre-projection query-head replacements for scored initialization."""
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



@contextmanager
def capture_pre(module, position, sink):
    def hook(m,args):sink.append(args[0][0,position].detach().cpu().clone())
    handle=module.register_forward_pre_hook(hook)
    try:yield
    finally:handle.remove()
    if len(sink)!=1:raise ValueError('Expected one pre-projection capture')

@contextmanager
def replace_heads(module,position,donor,heads,head_dim):
    calls=[0]
    def hook(m,args):
        if calls[0]:return args
        calls[0]+=1
        x=args[0]
        if donor.shape != x[0,position].shape:raise ValueError('Donor shape mismatch')
        changed=x.clone()
        for head in heads:
            if head<0 or (head+1)*head_dim>x.shape[-1]:raise ValueError('Head out of range')
            sl=slice(head*head_dim,(head+1)*head_dim)
            changed[:,position,sl]=donor[sl].to(changed)
        return (changed,)+args[1:]
    handle=module.register_forward_pre_hook(hook)
    try:yield
    finally:handle.remove()
    if calls[0]!=1:raise ValueError('Head replacement did not execute')

def choose_combination(rows, threshold):
    eligible=[r for r in rows if r['case']=='changed' and r['direction']=='reversed_to_forward' and r['kind']=='single' and r['control']=='donor' and r['margin_delta']>=threshold]
    eligible.sort(key=lambda r:(-r['margin_delta'],int(r['heads'])))
    return [int(r['heads']) for r in eligible[:2]] if len(eligible)>=2 else []

def plot_results():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((OUT/'head_effects_data.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,case in enumerate(('changed','unchanged')):
        for j,direction in enumerate(('forward_to_reversed','reversed_to_forward')):
            ax=axes[i,j];group=[r for r in rows if r['case']==case and r['direction']==direction]
            singles=[r for r in group if r['kind']=='single']
            ax.bar([int(r['heads']) for r in singles],[float(r['margin_delta']) for r in singles])
            for kind,color in [('all','orange'),('joint','purple')]:
                ref=next((r for r in group if r['kind']==kind),None)
                if ref:ax.axhline(float(ref['margin_delta']),color=color,ls='--',label=('all heads' if kind=='all' else 'joint heads: '+ref['heads']))
            ax.axhline(0,color='gray',lw=.6)
            ax.axhline(-float(group[0]['recipient_margin']),color='black',ls=':',label='choice boundary')
            ax.set(title=case+' / '+direction,xlabel='Query head (zero indexed)',ylabel='Margin change');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'head_effects.png',dpi=180);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=('cuda','mps','auto','cpu'),default='auto',help='Auto prefers CUDA then MPS; never falls back to CPU. CPU is explicit diagnostics only.');ap.add_argument('--tolerance',type=float,default=1e-4);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--plot-only',action='store_true');args=ap.parse_args()
    cfg=json.loads((OUT/'config.json').read_text())
    if cfg['procedure']!='scored' or cfg['layer']!=15 or cfg['combination_max_heads']!=2:
        raise ValueError('This pilot is configured for scored block-15 heads and at most two joint heads')
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
    layer=cfg['layer'];attention=runtime.model.model.layers[layer].self_attn;projection=attention.o_proj
    heads=runtime.model.config.num_attention_heads
    kv_heads=runtime.model.config.num_key_value_heads
    width=projection.in_features
    head_dim=getattr(attention,'head_dim',width//heads)
    if width!=heads*head_dim or heads%kv_heads:raise ValueError('Unsupported grouped query layout')
    save('head_layout.json',dict(layer=layer,query_heads=heads,key_value_heads=kv_heads,head_dim=head_dim,projection_input_width=width,target='weighted query-head outputs before o_proj'))
    reference_path=ROOT/cfg['reference']
    references=list(csv.DictReader(reference_path.open()))
    save('reference_provenance.json',dict(path=cfg['reference'],sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest()))
    captures=[];observations=[]
    for item,base in zip(inputs,replay):
        _,encoded=runtime.inputs(item['messages']);position=encoded.input_ids.shape[1]-1
        pre=[];post=[];scores=[]
        for k in (1,2):
            sink=[];out={}
            with capture_pre(projection,position,sink),observe([attention],position,out):
                scores.append(runtime.likelihood(item['messages'],item['scenario'][f'strategy_{k}']['candidate'])[0])
            pre.append(sink[0]);post.append(out[0])
        error=max(abs(scores[0]-base['likelihood_1']),abs(scores[1]-base['likelihood_2']))
        suffix_error=max((pre[0]-pre[1]).abs().max().item(),(post[0]-post[1]).abs().max().item())
        observations.append(dict(case=item['case'],direction=item['direction'],score_error=error,suffix_error=suffix_error,pass_check=error<=args.tolerance and suffix_error<=args.tolerance))
        table('observation_controls.csv',observations)
        if not observations[-1]['pass_check']:raise ValueError('Observation failed')
        torch.save(dict(position=position,pre_projection=pre[0],attention_output=post[0],heads=heads,head_dim=head_dim),OUT/f"activations_{item['case']}_{item['direction']}.pt")
        captures.append((position,pre[0],post[0]))
    results=[];location=[]
    def evaluate(recipient,donor,selected,kind,control):
        item=inputs[recipient];base=replay[recipient];target=replay[donor];source=recipient if control=='self' else donor
        scores=[]
        for k in (1,2):
            with replace_heads(projection,captures[recipient][0],captures[source][1],selected,head_dim):
                scores.append(runtime.likelihood(item['messages'],item['scenario'][f'strategy_{k}']['candidate'])[0])
        margin=scores[0]-scores[1];strategy=1 if margin>=0 else 2
        error=max(abs(scores[0]-base['likelihood_1']),abs(scores[1]-base['likelihood_2']))
        row=dict(case=item['case'],direction=inputs[donor]['direction']+'_to_'+item['direction'],layer=layer,kind=kind,heads=','.join(map(str,selected)),control=control,likelihood_1=scores[0],likelihood_2=scores[1],margin=margin,margin_delta=margin-base['margin'],recipient_margin=base['margin'],donor_margin=target['margin'],strategy=strategy,recipient_strategy=base['strategy'],donor_strategy=target['strategy'],transfer=int(base['strategy']!=target['strategy'] and strategy==target['strategy']),self_error=error if control=='self' else '')
        results.append(row);table('patches.csv',results)
        if control=='self' and (error>args.tolerance or strategy!=base['strategy']):raise ValueError('Self replacement failed')
        return row
    pairs=[(1,0),(0,1),(3,2),(2,3)]
    # Validate all directions BEFORE inspecting individual heads.
    for recipient,donor in pairs:
        for control in ('self','donor'):
            row=evaluate(recipient,donor,list(range(heads)),'all',control)
            source=recipient if control=='self' else donor;item=inputs[recipient];scores=[]
            for k in (1,2):
                with patch(attention,captures[recipient][0],captures[source][2]):
                    scores.append(runtime.likelihood(item['messages'],item['scenario'][f'strategy_{k}']['candidate'])[0])
            ref=next(r for r in references if r['case']==row['case'] and r['direction']==row['direction'] and r['component']=='attention' and int(r['layer'])==layer and r['control']==control)
            current_error=max(abs(row[f'likelihood_{k}']-scores[k-1]) for k in (1,2))
            prior_error=max(abs(row[f'likelihood_{k}']-float(ref[f'likelihood_{k}'])) for k in (1,2))
            location.append(dict(case=row['case'],direction=row['direction'],control=control,all_score_1=row['likelihood_1'],all_score_2=row['likelihood_2'],whole_attention_score_1=scores[0],whole_attention_score_2=scores[1],prior_score_1=ref['likelihood_1'],prior_score_2=ref['likelihood_2'],current_error=current_error,prior_error=prior_error,pass_check=max(current_error,prior_error)<=args.tolerance))
            table('location_controls.csv',location)
            if not location[-1]['pass_check']:raise ValueError('All-head/whole-attention equivalence failed; no individual head sweep')
    for recipient,donor in pairs:
        for head in range(heads):
            for control in ('self','donor'):evaluate(recipient,donor,[head],'single',control)
            print(inputs[recipient]['case'],inputs[recipient]['direction'],'head',head,flush=True)
    selected=choose_combination(results,cfg['combination_threshold'])
    save('combination_selection.json',dict(heads=selected,threshold=cfg['combination_threshold'],rule='Top two positive margin effects >= threshold in changed reversed_to_forward; same pair applied in every context',reason='two qualifying heads' if selected else 'fewer than two qualifying heads; joint test skipped'))
    if selected:
        for recipient,donor in pairs:
            for control in ('self','donor'):evaluate(recipient,donor,selected,'joint',control)
    table('head_effects_data.csv',[r for r in results if r['control']=='donor'])
    plot_results()
    summary=['# Head pilot results','', 'All replay, observation, self and all-head location controls passed.', 'Heads are query-head outputs before projection. Positive deltas favor S1.', '']
    for recipient,donor in pairs:
        direction=inputs[donor]['direction']+'_to_'+inputs[recipient]['direction'];case=inputs[recipient]['case']
        group=[r for r in results if r['case']==case and r['direction']==direction and r['kind']=='single' and r['control']=='donor']
        positive=max(group,key=lambda r:r['margin_delta']);negative=min(group,key=lambda r:r['margin_delta'])
        summary.append(f"- {case}, {direction}: largest signed delta {positive['margin_delta']:+.6f} at head {positive['heads']}; smallest {negative['margin_delta']:+.6f} at head {negative['heads']}. Transfer heads: {[r['heads'] for r in group if r['transfer']]}")
    summary+=['',f'Joint selection: {selected or "skipped"}. Joint scores are measured directly and saved in patches.csv; no additive prediction is substituted.', 'The selection threshold is exploratory, not statistical significance. Both cases and directions are needed for interpretation. A margin shift is not necessarily strategy transfer. These results do not prove head necessity or generalize beyond the selected scored initialization pairs.']
    (OUT/'interpretation.md').write_text('\n'.join(summary)+'\n')
    save('status.json',dict(status='complete',patch_rows=len(results),heads=heads))

if __name__=='__main__':
    try:main()
    except Exception as error:
        save('status.json',dict(status='failed',error=str(error)))
        raise
