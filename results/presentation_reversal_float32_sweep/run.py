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
    with (OUT/'patches.csv').open() as f:current=list(csv.DictReader(f))
    cfg=json.loads((OUT/'config.json').read_text())
    refpath=ROOT/cfg['reference']
    with refpath.open() as f:prior=list(csv.DictReader(f))
    prior=[r for r in prior if r['case']=='changed' and r['control']=='donor']
    current=[r for r in current if r['control']=='donor']
    comparison=[]
    for precision,rows in [('bfloat16',prior),('float32',current)]:
        for direction in ('forward_to_reversed','reversed_to_forward'):
            group=[r for r in rows if r['direction']==direction]
            if sorted(int(r['layer']) for r in group)!=list(range(28)):raise ValueError('Incomplete sweep')
            effects=[abs(float(r['margin_delta'])) for r in group]
            for r in group:
                delta=float(r['margin_delta']);margin=float(r['margin'])
                comparison.append(dict(precision=precision,direction=direction,layer=int(r['layer']),baseline_margin=margin-delta,margin=margin,margin_delta=delta,strategy=int(r['strategy']),recipient_strategy=int(r['recipient_strategy']),donor_strategy=int(r['donor_strategy']),rank_absolute_effect=1+sum(e>abs(delta) for e in effects),transfer=int(r['recipient_strategy']!=r['donor_strategy'] and r['strategy']==r['donor_strategy'])))
    table('precision_comparison.csv',comparison)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    limit=max(abs(r['margin_delta']) for r in comparison)*1.1 or 1
    for i,d in enumerate(('forward_to_reversed','reversed_to_forward')):
        for precision in ('bfloat16','float32'):
            rows=sorted([r for r in comparison if r['direction']==d and r['precision']==precision],key=lambda r:r['layer'])
            axes[i,0].plot([r['layer'] for r in rows],[r['margin_delta'] for r in rows],'.-',label=precision)
            axes[i,1].plot([r['layer'] for r in rows],[r['strategy'] for r in rows],'.-',label=precision)
        axes[i,0].set(title=d,ylabel='Margin change',ylim=(-limit,limit));axes[i,0].axhline(0,color='gray',lw=.5)
        axes[i,1].set(title='Resulting scored strategy',yticks=[1,2],ylim=(.8,2.2))
        for ax in axes[i]:ax.set_xlabel('Block (zero indexed)');ax.axvline(15,color='gray',ls=':');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'precision_sweep.png',dpi=180);plt.close(fig)
    lines=['# Float32 residual sweep comparison','','Ranks use absolute margin change within each precision and direction (ties share minimum rank). Baselines and donors are recomputed in float32. The reference includes native bfloat16 scoring, so this compares complete execution/scoring configurations.','','| Precision | Direction | Block 15 delta | Block 15 rank | Strongest block(s) | Strategy transfers |','|---|---|---:|---:|---|---|']
    for precision in ('bfloat16','float32'):
        for direction in ('forward_to_reversed','reversed_to_forward'):
            g=[r for r in comparison if r['precision']==precision and r['direction']==direction];r=next(r for r in g if r['layer']==15)
            lines.append(f"| {precision} | {direction} | {r['margin_delta']:+.9f} | {r['rank_absolute_effect']} | {[x['layer'] for x in g if x['rank_absolute_effect']==1]} | {[x['layer'] for x in g if x['transfer']]} |")
    lines+=['','Positive changes favor S1; donor-directed change depends on donor/recipient baselines. Use the signed changes in both directions to assess whether the original asymmetry persists. No necessity, significance or generalization claim follows from a single selected pair.']
    (OUT/'results_summary.md').write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cuda','mps','auto'],default='auto');ap.add_argument('--plot-only',action='store_true');args=ap.parse_args()
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
    import torch,transformers
    device=base_pilot.pilot_device(torch,args.device)
    save('status.json',dict(status='running'))
    if device=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime=FloatRuntime(baseline[0]['model'],device);runtime.model.float()
    assert all(p.dtype==torch.float32 for p in runtime.model.parameters() if p.is_floating_point())
    blocks=runtime.model.model.layers
    assert len(blocks)==28
    save('environment.json',dict(device=device,torch=torch.__version__,transformers=transformers.__version__,parameter_dtype=str(next(runtime.model.parameters()).dtype),logits_dtype='torch.float32 (asserted each call)',logsoftmax_dtype='torch.float32',accumulation='torch.float32 sum',tf32=False if device=='cuda' else None,revision=getattr(runtime.model.config,'_commit_hash',None)))
    refpath=ROOT/cfg['reference'];save('reference_provenance.json',dict(path=cfg['reference'],sha256=hashlib.sha256(refpath.read_bytes()).hexdigest()))
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
            checks.append(dict(presentation=item['direction'],candidate=k,observation_error=abs(score-observed)))
            table('observation_scores.csv',checks)
            if score!=observed:raise ValueError('Observation changed score')
            scores.append(score);vectors.append(captured)
        suffix_error=max((vectors[0][i]-vectors[1][i]).abs().max().item() for i in range(28))
        if suffix_error!=0:raise ValueError('Candidate suffix affected prompt activation')
        margin=scores[0]-scores[1]
        baselines.append(dict(presentation=item['direction'],likelihood_1=scores[0],likelihood_2=scores[1],margin=margin,strategy=1 if margin>=0 else 2,saved_strategy=item['row']['strategy'],saved_margin=item['row']['decision']['margin'],suffix_activation_error=suffix_error))
        table('float32_baselines.csv',baselines)
        torch.save(dict(position=position,hidden_states=vectors[0]),OUT/f"activations_{item['direction']}.pt")
        captures.append((position,vectors[0]))
    results=[]
    for recipient,donor in [(1,0),(0,1)]:
        item=inputs[recipient];baseline=baselines[recipient];target=baselines[donor]
        for layer,block in enumerate(blocks):
            for control,source in [('self',recipient),('donor',donor)]:
                scores=[]
                for k in (1,2):
                    with base_pilot.patch(block,captures[recipient][0],captures[source][1][layer]):
                        score,_,_=runtime.likelihood(item['messages'],item['scenario'][f'strategy_{k}']['candidate'])
                    scores.append(score)
                margin=scores[0]-scores[1];strategy=1 if margin>=0 else 2
                error=max(abs(scores[k-1]-baseline[f'likelihood_{k}']) for k in (1,2))
                results.append(dict(case='changed',direction=inputs[donor]['direction']+'_to_'+item['direction'],layer=layer,control=control,likelihood_1=scores[0],likelihood_2=scores[1],margin=margin,margin_delta=margin-baseline['margin'],strategy=strategy,recipient_strategy=baseline['strategy'],donor_strategy=target['strategy'],original_saved_strategy=item['row']['strategy'],self_score_error=error if control=='self' else ''))
                table('patches.csv',results)
                if control=='self' and error!=0:raise ValueError('Self replacement mismatch')
            print(results[-1]['direction'],'block',layer,flush=True)
    summarize();save('status.json',dict(status='complete',patch_rows=len(results)))

if __name__=='__main__':
    try:main()
    except Exception as e:save('status.json',dict(status='failed',error=str(e)));raise
