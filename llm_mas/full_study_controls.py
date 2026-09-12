"""Separate message/history controls over saved natural synchronous contexts."""
import argparse
import json
import hashlib
import gc
from pathlib import Path
from .config import ROOT, digest
from .prompts import for_condition
from .runner import write
from .full_study_analysis import table


def branches(row):
    deliveries=row['deliveries'];base=row['decision_history'][:-len(deliveries)]
    for history_mode in ['full','no_self']:
        history=base if history_mode=='full' else [m for m in base if m['role']!='assistant']
        for component in ['full','endorsement','explanation','none']:
            incoming=[]
            if component!='none':
                for d in deliveries:
                    message=d['message']
                    # Saved structured generated messages; forced/scored messages have a fixed first line.
                    from .choice_output import parse
                    strategy, explanation, status = parse(message)
                    if status == 'valid':
                        endorsement=f'Strategy {strategy}'
                    else:
                        endorsement,_,explanation=message.partition('\n')
                    text=message if component=='full' else endorsement if component=='endorsement' else explanation
                    incoming.append({'role':'user','content':f'{d["speaker"]} said:\n{text}'})
            yield history_mode,component,history+incoming


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda']);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    source=a.source.resolve();out=a.output.resolve()
    if out==source or source in out.parents:p.error('Use output outside source run')
    records=[(f.stem,json.loads(f.read_text())) for f in sorted((source/'conversations').glob('*.json'))]
    events=[(t,d,r) for t,d in records if not any(r.get('choice_status') in ('invalid_json','invalid_schema') for r in d['rows']) and d['config']['initialization']=='natural' and d['config']['schedule']=='synchronous' for r in d['rows'] if r['round'] in (1,2) and r['deliveries']]
    if not events:p.error('No eligible natural synchronous contexts')
    print(f'{len(events)} contexts, {len(events)*8} branches')
    if a.dry_run:return
    templates=json.loads((source/'inputs/prompts.json').read_text())
    identity=dict(source=str(source),records_hash=digest(records),prompts_hash=digest(templates),device=a.device,script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=identity:p.error('Output inputs changed')
    write(out/'manifest.json',identity)
    from .runtime import Runtime
    summaries=[]
    for model in sorted({r['model'] for _,_,r in events}):
        runtime=None
        for task,d,row in events:
            if row['model']!=model:continue
            texts=for_condition(templates,d['config'],d['scenario'])
            for hmode,component,history in branches(row):
                target=out/'records'/f'{task}_{row["agent"].replace(" ","_")}_{row["round"]}_{hmode}_{component}.json'
                if target.exists():result=json.loads(target.read_text())
                else:
                    if runtime is None:runtime=Runtime(model,a.device)
                    score=runtime.measure(history,d['scenario'],texts)
                    result=dict(task=task,agent=row['agent'],round=row['round'],model=model,history_mode=hmode,component=component,history=history,measurement=score)
                    write(target,result)
                summaries.append({k:result[k] for k in ['task','agent','round','model','history_mode','component']}|{'margin':result['measurement']['margin']})
        del runtime;gc.collect()
    baseline={(r['task'],r['agent'],r['round'],r['history_mode']):r['margin'] for r in summaries if r['component']=='full'}
    for r in summaries:r['delta_from_full_message']=r['margin']-baseline[(r['task'],r['agent'],r['round'],r['history_mode'])]
    table(out/'controls.csv',summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len({r['model'] for r in summaries}),figsize=(14,5),squeeze=False,layout='constrained')
    for ax,model in zip(axes[0],sorted({r['model'] for r in summaries})):
        labels=[];values=[]
        for hm in ['full','no_self']:
            for comp in ['endorsement','explanation','none']:
                labels.append(hm+' / '+comp);values.append([r['delta_from_full_message'] for r in summaries if r['model']==model and r['history_mode']==hm and r['component']==comp])
        ax.boxplot(values);ax.set_xticks(range(1,7),labels,rotation=75);ax.axhline(0,color='gray');ax.set_title(model.split('/')[-1]);ax.set_ylabel('Margin difference from full message')
    for ext in ['png','pdf']:fig.savefig(out/f'controls.{ext}',bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
