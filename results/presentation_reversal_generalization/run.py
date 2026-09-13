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

import importlib.util
spec=importlib.util.spec_from_file_location('base_pilot',ROOT/'results/presentation_reversal_patching/run.py')
base_pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(base_pilot)

class FloatRuntime(Runtime):
    def likelihood(self,messages,candidate,continuation_prefix=''):
        torch=self.torch
        rendered,encoded=self.inputs(messages,continuation_prefix+candidate)
        rendered+=continuation_prefix
        prefix=self.encode_rendered(rendered)['input_ids'];start=len(prefix)
        if encoded.input_ids[0,:start].tolist()!=prefix:raise ValueError('Candidate boundary mismatch')
        with torch.inference_mode():
            output=self.model(**encoded,use_cache=False)
            logits=output.logits[:,start-1:encoded.input_ids.shape[1]-1,:]
            if logits.dtype!=torch.float32:raise ValueError('Expected float32 logits')
            lp=torch.log_softmax(logits.float(),dim=-1)
            gathered=lp.gather(-1,encoded.input_ids[:,start:].unsqueeze(-1)).squeeze(-1)[0]
            score=gathered.sum(dtype=torch.float32).item();values=gathered.tolist()
        return score,values,rendered

def summarize():
    with (OUT/'patches.csv').open() as f:rows=list(csv.DictReader(f))
    donors=[r for r in rows if r['control']=='donor']
    table('figure_data.csv',donors)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=list(dict.fromkeys(r['case'] for r in donors))
    fig,axes=plt.subplots(len(cases),2,figsize=(12,3*len(cases)),squeeze=False)
    limit=max(abs(float(r['margin_delta'])) for r in donors)*1.1 or 1
    lines=['# Float32 replacement across distinct initialization pairs','','Directions identify donor → recipient. Positive margin changes favor the semantic S1 alternative; negative changes favor S2. Saved classes are selection labels; float32 baselines are recomputed. No head searches were run.','','| Case | Group | Direction | Block | Baseline | Delta | Result | Semantic choice |','|---|---|---|---:|---:|---:|---|---|']
    for i,case in enumerate(cases):
        for direction in ('forward_to_reversed','reversed_to_forward'):
            g=sorted([r for r in donors if r['case']==case and r['direction']==direction],key=lambda r:int(r['layer']))
            if [int(r['layer']) for r in g]!=[12,13,14,15]:raise ValueError('Incomplete sweep')
            axes[i,0].plot([int(r['layer']) for r in g],[float(r['margin_delta']) for r in g],'.-',label=direction)
            axes[i,1].plot([int(r['layer']) for r in g],[int(r['strategy']) for r in g],'.-',label=direction)
            for r in g:lines.append(f"| {case} | {r['group']} | {direction} | {r['layer']} | {float(r['baseline_margin']):+.6f} | {float(r['margin_delta']):+.6f} | S{r['strategy']} | {r['semantic_strategy']} |")
        axes[i,0].set(title=case+' ('+g[0]['group']+')',ylabel='Margin change',ylim=(-limit,limit));axes[i,0].axhline(0,color='gray',lw=.5)
        axes[i,1].set(title='Resulting strategy',yticks=[1,2],yticklabels=['S1: '+g[0]['semantic_s1'],'S2: '+g[0]['semantic_s2']],ylim=(.8,2.2))
        for ax in axes[i]:ax.set_xticks([12,13,14,15]);ax.set_xlabel('Block (zero indexed)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'case_effects.png',dpi=180);plt.close(fig)
    (OUT/'results_summary.md').write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cuda','mps','auto'],default='auto');ap.add_argument('--plot-only',action='store_true');ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    if args.plot_only:
        if json.loads((OUT/'status.json').read_text())['status']!='complete':raise ValueError('Require completed sweep')
        summarize();return
    cfg=json.loads((OUT/'config.json').read_text())
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
            inputs.append(dict(case=case['name'],group=case['group'],saved_class=case['saved_class'],direction=direction,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),config=c,scenario=scenario,row=row,texts=texts,history=history,messages=history+[dict(role='user',content=probe)]))
            baseline.append(dict(case=case['name'],group=case['group'],saved_class=case['saved_class'],direction=direction,agent=case['agent'],model=row['model'],scenario=scenario['id'],persona=c.get('persona',False),initialization=c['initialization'],procedure=c['decision_mode'],schedule=c['schedule'],strategy=row['strategy'],likelihood_1=row['decision']['likelihoods'][0],likelihood_2=row['decision']['likelihoods'][1],margin=row['decision']['margin'],message=row['message'],source=str(path.relative_to(ROOT))))
    # Require identical conditions except the two presentation manipulations.
    for f,r in zip(inputs[::2],inputs[1::2]):
        assert f['scenario']==r['scenario']
        for key in ('models','persona','initialization','decision_mode','schedule','argument_mode','max_argument_tokens','decoding'):
            assert f['config'].get(key)==r['config'].get(key),key
        assert not f['config'].get('reverse_candidate_blocks',False) and r['config']['reverse_candidate_blocks']
        assert f['config']['presentation_order']=='strategy_1_first' and r['config']['presentation_order']=='strategy_2_first'
    hashes=[]
    for case,f,r in zip(cfg['cases'],inputs[::2],inputs[1::2]):
        h=hashlib.sha256((f['row']['decision']['rendered_prompt']+'\0'+r['row']['decision']['rendered_prompt']).encode()).hexdigest()
        assert h==case['prompt_pair_sha256']
        hashes.append(h)
    assert len(set(hashes))==len(hashes), 'Duplicate initialization prompts'
    table('matched_cases.csv',baseline);save('inputs.json',inputs)
    if args.prepare_only:return
    import torch,transformers
    device=base_pilot.pilot_device(torch,args.device)
    save('status.json',dict(status='running'))
    if device=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime=FloatRuntime(baseline[0]['model'],device);runtime.model.float()
    assert all(p.dtype==torch.float32 for p in runtime.model.parameters() if p.is_floating_point())
    blocks=runtime.model.model.layers
    assert len(blocks)==28
    save('environment.json',dict(device=device,torch=torch.__version__,transformers=transformers.__version__,parameter_dtype=str(next(runtime.model.parameters()).dtype),logits_dtype='torch.float32 (asserted each call)',logsoftmax_dtype='torch.float32',accumulation='torch.float32 sum',tf32=False if device=='cuda' else None,revision=getattr(runtime.model.config,'_commit_hash',None)))
    save('configuration_used.json',cfg)
    baselines=[];captures=[];checks=[]
    for item in inputs:
        _,encoded=runtime.inputs(item['messages']);position=encoded.input_ids.shape[1]-1
        assert runtime.inputs(item['messages'])[0]==item['row']['decision']['rendered_prompt']
        scores=[];vectors=[]
        for k in (1,2):
            candidate=item['scenario'][f'strategy_{k}']['candidate'];score,_,_=runtime.likelihood(item['messages'],candidate)
            captured={}
            with base_pilot.observe(blocks,position,captured):observed,_,_=runtime.likelihood(item['messages'],candidate)
            assert len(captured)==28
            checks.append(dict(case=item['case'],presentation=item['direction'],candidate=k,observation_error=abs(score-observed)))
            table('observation_scores.csv',checks)
            if score!=observed:raise ValueError('Observation changed score')
            scores.append(score);vectors.append(captured)
        suffix_error=max((vectors[0][i]-vectors[1][i]).abs().max().item() for i in range(28))
        if suffix_error!=0:raise ValueError('Candidate suffix affected prompt activation')
        margin=scores[0]-scores[1]
        baselines.append(dict(case=item['case'],group=item['group'],saved_class=item['saved_class'],semantic_s1=item['scenario']['strategy_1']['name'],semantic_s2=item['scenario']['strategy_2']['name'],presentation=item['direction'],likelihood_1=scores[0],likelihood_2=scores[1],margin=margin,strategy=1 if margin>=0 else 2,saved_strategy=item['row']['strategy'],saved_margin=item['row']['decision']['margin'],suffix_activation_error=suffix_error))
        table('float32_baselines.csv',baselines)
        torch.save(dict(position=position,hidden_states=vectors[0]),OUT/f"activations_{item['case']}_{item['direction']}.pt")
        captures.append((position,vectors[0]))
    results=[]
    for recipient,donor in [(j+a,j+b) for j in range(0,len(inputs),2) for a,b in [(1,0),(0,1)]]:
        item=inputs[recipient];baseline=baselines[recipient];target=baselines[donor]
        for layer in cfg['blocks']:
            block=blocks[layer]
            for control,source in [('self',recipient),('donor',donor)]:
                scores=[]
                for k in (1,2):
                    with base_pilot.patch(block,captures[recipient][0],captures[source][1][layer]):
                        score,_,_=runtime.likelihood(item['messages'],item['scenario'][f'strategy_{k}']['candidate'])
                    scores.append(score)
                margin=scores[0]-scores[1];strategy=1 if margin>=0 else 2
                error=max(abs(scores[k-1]-baseline[f'likelihood_{k}']) for k in (1,2))
                results.append(dict(case=item['case'],group=item['group'],saved_class=item['saved_class'],semantic_s1=item['scenario']['strategy_1']['name'],semantic_s2=item['scenario']['strategy_2']['name'],semantic_strategy=item['scenario'][f'strategy_{strategy}']['name'],baseline_margin=baseline['margin'],baseline_likelihood_1=baseline['likelihood_1'],baseline_likelihood_2=baseline['likelihood_2'],direction=inputs[donor]['direction']+'_to_'+item['direction'],layer=layer,control=control,likelihood_1=scores[0],likelihood_2=scores[1],margin=margin,margin_delta=margin-baseline['margin'],strategy=strategy,recipient_strategy=baseline['strategy'],donor_strategy=target['strategy'],original_saved_strategy=item['row']['strategy'],self_score_error=error if control=='self' else ''))
                table('patches.csv',results)
                if control=='self' and error!=0:raise ValueError('Self replacement mismatch')
            print(results[-1]['direction'],'block',layer,flush=True)
    summarize();save('status.json',dict(status='complete',patch_rows=len(results)))

if __name__=='__main__':
    try:main()
    except Exception as e:save('status.json',dict(status='failed',error=str(e)));raise
