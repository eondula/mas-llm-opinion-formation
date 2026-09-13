"""Scored initialization: attention/MLP outputs and residual reference, blocks 12–15."""
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


def component_modules(runtime, layers):
    blocks=runtime.model.model.layers
    modules=[]
    for layer in layers:
        if not 0 <= layer < len(blocks):raise ValueError(f'Invalid block {layer}')
        block=blocks[layer]
        # Attention module output is after o_proj, before residual addition.
        # MLP module output is after down_proj, before residual addition.
        for component,module in [('attention',block.self_attn),('mlp',block.mlp),('residual',block)]:
            modules.append((layer,component,module))
    return modules


def export_figure_data(results):
    fields=('case','direction','layer','component','control','likelihood_1','likelihood_2',
            'margin','recipient_margin','donor_margin','margin_delta','strategy',
            'recipient_strategy','donor_strategy','transfer')
    rows=[]
    for result in results:
        if result['control']!='donor':continue
        row={key:result[key] for key in fields}
        row['choice_boundary_delta']=-float(result['recipient_margin'])
        rows.append(row)
    table('component_effects_data.csv',rows)


def plot_results(cfg):
    # Both initial plots and regeneration use the exact same saved CSV.
    with (OUT/'component_effects_data.csv').open() as source:
        results=list(csv.DictReader(source))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,12),squeeze=False)
    for i,case in enumerate(cfg['cases']):
        for j,direction in enumerate(('forward_to_reversed','reversed_to_forward')):
            a,b=axes[2*i+j]
            for component in ('attention','mlp','residual'):
                rows=sorted([r for r in results if r['case']==case['name'] and r['direction']==direction and r['component']==component and r['control']=='donor'],key=lambda r:int(r['layer']))
                a.plot([int(r['layer']) for r in rows],[float(r['margin_delta']) for r in rows],'.-',label=component)
                b.plot([int(r['layer']) for r in rows],[int(r['strategy']) for r in rows],'.-',label=component)
            a.axhline(0,color='gray',lw=.7)
            reference=rows[0]
            a.axhline(float(reference['choice_boundary_delta']),color='black',ls=':',lw=1,label='choice boundary')
            a.set(title=case['name']+' / '+direction,ylabel='Patched minus recipient margin')
            b.set(title='Selected strategy',ylabel='Semantic strategy',yticks=[1,2],ylim=(.8,2.2))
            for ax in (a,b):ax.set_xlabel('Block (zero indexed)');ax.set_xticks(cfg['layers']);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'component_effects.png',dpi=180);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=('cuda','mps','auto','cpu'),default='auto',help='Auto prefers CUDA then MPS; never falls back to CPU. CPU is explicit diagnostics only.');ap.add_argument('--tolerance',type=float,default=1e-4);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--plot-only',action='store_true');args=ap.parse_args()
    cfg=json.loads((OUT/'config.json').read_text())
    if cfg.get('procedure')!='scored' or cfg.get('components')!=['attention','mlp','residual']:
        raise ValueError('This pilot supports scored attention/MLP/residual interventions only')
    if not cfg['layers'] or len(set(cfg['layers']))!=len(cfg['layers']):
        raise ValueError('Layers must be a nonempty list of unique block indices')
    if args.plot_only:
        if json.loads((OUT/'status.json').read_text()).get('status')!='complete':
            raise ValueError('Plot-only requires a completed, verified sweep')
        rows=list(csv.DictReader((OUT/'patches.csv').open()))
        for row in rows:
            for key in ('layer','strategy'):row[key]=int(row[key])
            row['margin_delta']=float(row['margin_delta'])
        export_figure_data(rows)
        plot_results(cfg)
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
    modules=component_modules(runtime,cfg['layers']);captures=[];controls=[]
    for item,base in zip(inputs,replay):
        _,encoded=runtime.inputs(item['messages']);position=encoded.input_ids.shape[1]-1
        vectors={};other={}
        with observe([m for _,_,m in modules],position,vectors):
            first=runtime.likelihood(item['messages'],item['scenario']['strategy_1']['candidate'])[0]
        with observe([m for _,_,m in modules],position,other):
            second=runtime.likelihood(item['messages'],item['scenario']['strategy_2']['candidate'])[0]
        if set(vectors)!=set(range(len(modules))) or set(other)!=set(vectors):
            raise ValueError('Some observation hooks did not execute')
        err=max(abs(first-base['likelihood_1']),abs(second-base['likelihood_2']))
        for index,(layer,component,_) in enumerate(modules):
            causal_error=(vectors[index]-other[index]).abs().max().item()
            controls.append(dict(case=item['case'],direction=item['direction'],layer=layer,component=component,score_error=err,candidate_suffix_activation_error=causal_error,pass_check=err<=args.tolerance and causal_error<=args.tolerance))
        table('observation_controls.csv',controls)
        if not all(r['pass_check'] for r in controls):raise ValueError('Observation control failed')
        torch.save(dict(position=position,modules=[dict(layer=l,component=c) for l,c,_ in modules],hidden_states=vectors),OUT/f"activations_{item['case']}_{item['direction']}.pt")
        captures.append((position,vectors))
    save('status.json',dict(status='running',phase='replacement'))
    results=[]
    for start in range(0,len(inputs),2):
        for recipient,donor in ((start+1,start),(start,start+1)):
            item=inputs[recipient];base=replay[recipient];target=replay[donor]
            position,_=captures[recipient]
            for index,(layer,component,module) in enumerate(modules):
                for control,source in (('self',recipient),('donor',donor)):
                    scores=[]
                    # Separate contexts and no cache for every complete endorsement.
                    for k in (1,2):
                        with patch(module,position,captures[source][1][index]):
                            scores.append(runtime.likelihood(item['messages'],item['scenario'][f'strategy_{k}']['candidate'])[0])
                    margin=scores[0]-scores[1];strategy=1 if margin>=0 else 2
                    error=max(abs(scores[0]-base['likelihood_1']),abs(scores[1]-base['likelihood_2']))
                    results.append(dict(case=item['case'],direction=inputs[donor]['direction']+'_to_'+item['direction'],layer=layer,component=component,control=control,likelihood_1=scores[0],likelihood_2=scores[1],margin=margin,recipient_margin=base['margin'],donor_margin=target['margin'],margin_delta=margin-base['margin'],strategy=strategy,recipient_strategy=base['strategy'],donor_strategy=target['strategy'],matches_donor=int(strategy==target['strategy']),changed_strategy=int(strategy!=base['strategy']),transfer=int(base['strategy']!=target['strategy'] and strategy==target['strategy']),self_score_error=error if control=='self' else '',position=position))
                    table('patches.csv',results)
                    if control=='self' and (error>args.tolerance or strategy!=base['strategy']):raise ValueError('Self replacement failed; stop interpretation')
                print(item['case'],results[-1]['direction'],'block',layer,component,flush=True)
    # Aggregation preserves cases, directions and components; no pooled success rate.
    summaries=[]
    for case in cfg['cases']:
        for direction in ('forward_to_reversed','reversed_to_forward'):
            for component in ('attention','mlp','residual'):
                selected=[r for r in results if r['case']==case['name'] and r['direction']==direction and r['component']==component and r['control']=='donor']
                peak=max(selected,key=lambda r:abs(r['margin_delta']))
                summaries.append(dict(case=case['name'],direction=direction,component=component,peak_layer=peak['layer'],peak_delta=peak['margin_delta'],peak_margin=peak['margin'],transfer_layers=','.join(str(r['layer']) for r in selected if r['transfer']),choice_change_layers=','.join(str(r['layer']) for r in selected if r['changed_strategy'])))
    table('component_summary.csv',summaries)
    export_figure_data(results)
    plot_results(cfg)
    lines=['# Component pilot summary','',
           'All replay, observation and self controls passed. Layers are zero indexed.',
           'Each replacement is separate; attention and MLP effects are not additive.',
           'Attention is patched after its output projection and before residual addition; MLP is patched after its output projection and before residual addition.',
           '', '| Case | Direction | Component | Largest absolute effect: block | Margin change | Transfer blocks |',
           '|---|---|---|---:|---:|---|']
    for r in summaries:lines.append(f"| {r['case']} | {r['direction']} | {r['component']} | {r['peak_layer']} | {r['peak_delta']:+.6f} | {r['transfer_layers'] or 'none'} |")
    lines+=['','The unchanged pair cannot exhibit transfer because its baselines agree. Compare its margin sensitivity with the changed pair before selecting finer interventions.',
            'A concentrated attention effect supports a subsequent head-output test; an MLP effect supports finer MLP investigation. This selected four-block scan alone cannot establish a unique mechanism or justify treating one component as necessary.',
            'The residual reference is rerun in this environment. Stronger component effects do not imply additive contributions or mediation fractions. No generated-choice or conversation continuation is tested.']
    (OUT/'interpretation.md').write_text('\n'.join(lines)+'\n')
    save('status.json',dict(status='complete',patch_rows=len(results)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        save('status.json',dict(status='failed',error=str(error)))
        raise
