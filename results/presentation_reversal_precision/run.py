"""Changed reversed-to-forward only: separate scoring precision from forward dtype."""
import argparse,csv,hashlib,importlib.util,json,sys
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
from llm_mas.runtime import Runtime
spec=importlib.util.spec_from_file_location('heads',ROOT/'results/presentation_reversal_heads/run.py');heads=importlib.util.module_from_spec(spec);spec.loader.exec_module(heads)
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2)+'\n')
def table(n,rows):
    if rows:
        with (OUT/n).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def reductions(torch,logits,ids):
    results=[]
    for method in ('existing','float32_scoring'):
        lp=torch.log_softmax(logits if method=='existing' else logits.float(),dim=-1)
        gathered=lp.gather(-1,ids.unsqueeze(-1)).squeeze(-1)[0]
        values=gathered.tolist()
        if method=='existing':score=sum(values);accum='Python float (binary64); sum(list[float])'
        else:score=gathered.sum(dtype=torch.float32).item();accum='torch.float32 sum; Python float on export'
        results.append(dict(method=method,score=score,tokens=values,logits_dtype=str(logits.dtype),logsoftmax_dtype=str(lp.dtype),gathered_dtype=str(gathered.dtype),accumulation_dtype=accum,python_token_type=type(values[0]).__name__))
    return results

def score_both(runtime,messages,candidate):
    torch=runtime.torch
    rendered,encoded=runtime.inputs(messages,candidate)
    start=len(runtime.encode_rendered(rendered)['input_ids'])
    assert encoded.input_ids[0,:start].tolist()==runtime.encode_rendered(rendered)['input_ids']
    with torch.inference_mode():
        output=runtime.model(**encoded,use_cache=False)
        logits=output.logits[:,start-1:encoded.input_ids.shape[1]-1,:].clone()
        del output
        ids=encoded.input_ids[:,start:]
        result=reductions(torch,logits,ids)
    return result,ids[0].tolist()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['auto','cuda','mps'],default='auto');args=ap.parse_args()
    import torch,transformers
    device=heads.pilot_device(torch,args.device)
    save('status.json',dict(status='running'))
    path=ROOT/'results/presentation_reversal_head_confirmation/inputs.json'
    items=[x for x in json.loads(path.read_text()) if x['case']=='changed']
    donor=next(x for x in items if x['direction']=='reversed');recipient=next(x for x in items if x['direction']=='forward')
    assert recipient['row']['model']==donor['row']['model']=='Qwen/Qwen2.5-1.5B-Instruct'
    save('config.json',dict(case='changed',direction='reversed_to_forward',layer=15,heads=[2,9],source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),conditions=['baseline','head_2','head_9','joint'],forward_dtypes=['checkpoint_native','float32'],reductions=['existing','float32_scoring']))
    runtime=Runtime(recipient['row']['model'],device)
    # Disable reduced-mantissa CUDA float32 matmul. Record the settings explicitly.
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    save('environment.json',dict(torch=torch.__version__,transformers=transformers.__version__,device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else 'Apple MPS',native_parameter_dtype=str(next(runtime.model.parameters()).dtype),model_revision=getattr(runtime.model.config,'_commit_hash',None),cuda_tf32=False if device=='cuda' else None,forward_autocast=False,float32_weights='checkpoint weights converted to float32; no recovery of checkpoint quantization'))
    rows=[];token_rows=[];controls=[]
    for forward_dtype in ('checkpoint_native','float32'):
        if forward_dtype=='float32':runtime.model.float()
        parameter_dtype=str(next(runtime.model.parameters()).dtype)
        if forward_dtype=='float32':assert all(p.dtype==torch.float32 for p in runtime.model.parameters() if p.is_floating_point())
        projection=runtime.model.model.layers[15].self_attn.o_proj
        n=runtime.model.config.num_attention_heads;dim=projection.in_features//n
        assert projection.in_features==n*dim
        captures=[]
        for item in (recipient,donor):
            rendered,encoded=runtime.inputs(item['messages']);position=encoded.input_ids.shape[1]-1
            assert rendered==item['row']['decision']['rendered_prompt']
            candidate=item['scenario']['strategy_1']['candidate']
            baseline,_=score_both(runtime,item['messages'],candidate)
            sink=[]
            with heads.capture_pre(projection,position,sink):observed,_=score_both(runtime,item['messages'],candidate)
            assert all(a['score']==b['score'] for a,b in zip(baseline,observed))
            captures.append((position,sink[0]))
            torch.save(dict(position=position,vector=sink[0],parameter_dtype=parameter_dtype),OUT/f"activation_{forward_dtype}_{item['direction']}.pt")
        baselines={}
        for condition,selected in [('baseline',[]),('head_2',[2]),('head_9',[9]),('joint',[2,9])]:
            for control in (['none'] if not selected else ['self','donor']):
                scores=[];ids=[]
                for k in (1,2):
                    context=heads.replace_heads(projection,captures[0][0],captures[0 if control=='self' else 1][1],selected,dim) if selected else nullcontext()
                    with context:result,token_ids=score_both(runtime,recipient['messages'],recipient['scenario'][f'strategy_{k}']['candidate'])
                    scores.append(result);ids.append(token_ids)
                divergence=next(i for i,(a,b) in enumerate(zip(*ids)) if a!=b)
                for method_index,method in enumerate(('existing','float32_scoring')):
                    a,b=scores[0][method_index],scores[1][method_index]
                    margin=a['score']-b['score'];first=a['tokens'][divergence]-b['tokens'][divergence]
                    if condition=='baseline':baselines[method]=(margin,first)
                    row=dict(forward_dtype=forward_dtype,parameter_dtype=parameter_dtype,method=method,condition=condition,control=control,likelihood_1=a['score'],likelihood_2=b['score'],full_margin=margin,baseline_margin=baselines[method][0],margin_delta=margin-baselines[method][0],first_token_1=runtime.tokenizer.convert_ids_to_tokens(ids[0][divergence]),first_token_2=runtime.tokenizer.convert_ids_to_tokens(ids[1][divergence]),first_score_1=a['tokens'][divergence],first_score_2=b['tokens'][divergence],first_difference=first,baseline_first_difference=baselines[method][1],first_delta=first-baselines[method][1],original_saved_strategy=recipient['row']['strategy'],diagnostic_scored_choice=1 if margin>=0 else 2,logits_dtype=a['logits_dtype'],logsoftmax_dtype=a['logsoftmax_dtype'],gathered_dtype=a['gathered_dtype'],accumulation_dtype=a['accumulation_dtype'],python_token_type=a['python_token_type'])
                    rows.append(row)
                    for k,data in enumerate((a,b),1):
                        for i,(tid,value) in enumerate(zip(ids[k-1],data['tokens'])):token_rows.append(dict(forward_dtype=forward_dtype,method=method,condition=condition,control=control,candidate=k,token_index=i,token_id=tid,token=runtime.tokenizer.convert_ids_to_tokens(tid),logprob=value))
                    if control=='self':
                        error=abs(row['margin_delta']);first_error=abs(row['first_delta'])
                        controls.append(dict(forward_dtype=forward_dtype,method=method,condition=condition,margin_error=error,first_error=first_error,passed=error==0 and first_error==0))
                        if error or first_error:raise ValueError('Self replacement mismatch')
                table('precision_scores.csv',rows);table('token_scores.csv',token_rows);table('self_controls.csv',controls)
                print(forward_dtype,condition,control,flush=True)
        # Score donor baseline separately in each forward precision, using both reductions.
        ds=[score_both(runtime,donor['messages'],donor['scenario'][f'strategy_{k}']['candidate'])[0] for k in (1,2)]
        save(f'donor_baseline_{forward_dtype}.json',dict(parameter_dtype=parameter_dtype,scoring=ds))
    summaries=[]
    for precision in ('checkpoint_native','float32'):
        for method in ('existing','float32_scoring'):
            selected={r['condition']:r for r in rows if r['forward_dtype']==precision and r['method']==method and r['control'] in ('none','donor')}
            summaries.append(dict(forward_dtype=precision,method=method,full_joint_departure=selected['joint']['full_margin']-selected['head_2']['full_margin']-selected['head_9']['full_margin']+selected['baseline']['full_margin'],first_joint_departure=selected['joint']['first_difference']-selected['head_2']['first_difference']-selected['head_9']['first_difference']+selected['baseline']['first_difference']))
    table('joint_departures.csv',summaries)
    native=next(r for r in rows if r['forward_dtype']=='checkpoint_native' and r['method']=='existing' and r['condition']=='baseline')
    save('saved_baseline_comparison.json',dict(likelihood_1_error=native['likelihood_1']-recipient['row']['decision']['likelihoods'][0],likelihood_2_error=native['likelihood_2']-recipient['row']['decision']['likelihoods'][1],note='Dtype comparisons use their own recaptured donors and baselines; saved differences are reported, not used to force float32 replay.'))
    lines=['# Precision audit','','Same-logit scoring methods share a single forward call for each condition and candidate. Float32 forward uses converted checkpoint weights and recaptured donors. No cross-precision donor replacement.', '', '| Forward | Reduction | Full joint departure | First-difference joint departure |','|---|---|---:|---:|']
    for r in summaries:lines.append(f"| {r['forward_dtype']} | {r['method']} | {r['full_joint_departure']:+.9f} | {r['first_joint_departure']:+.9f} |")
    lines+=['','See precision_scores.csv for actual logits/log-softmax/gathered/accumulation dtypes, each baseline, first-difference scores, and all condition margins. Original saved strategy remains separate from diagnostic scored choice. No conversation or generated-choice revision is claimed.']
    (OUT/'results_summary.md').write_text('\n'.join(lines)+'\n');save('status.json',dict(status='complete'))
if __name__=='__main__':
    try:main()
    except Exception as e:save('status.json',dict(status='failed',error=str(e)));raise
