"""Identical-history probes, repeatability and presentation-order diagnostics."""
import math
from copy import deepcopy
from pathlib import Path
from .config import digest
from .prompts import question_initial
from .storage import write
from .behavioral import csv_write,average


def audit(path,tasks,texts,runtimes,protocol_hash,progress=lambda done,total:None,tolerance=1e-4):
    path=Path(path);rows=[];total=len(tasks)*3*2*2*len(runtimes)
    for task in tasks:
        base=question_initial(task,texts,2,'Agent A','synchronous')
        for context in ['question_only','stated_yes','stated_no']:
            history=deepcopy(base)
            if context!='question_only':history.append(dict(role='assistant',content=('Yes' if context=='stated_yes' else 'No')+'\nThis is my current position on the question.'))
            history_hash=digest(history)
            for order in ['yes_first','no_first']:
                variant=texts if order=='yes_first' else dict(texts,qa_probe=texts['qa_probe'].replace('{candidate_1}','{swapped}').replace('{candidate_2}','{candidate_1}').replace('{swapped}','{candidate_2}'))
                for name,runtime in runtimes.items():
                    previous=None
                    for repeat in range(2):
                        runtime.interaction_context=dict(stage='stance_audit',question_id=task['id'],context=context)
                        measure=runtime.measure(history,task,variant)
                        finite=all(math.isfinite(v) for v in [measure['margin'],measure['stance'],*measure['likelihoods']])
                        row=dict(question_id=task['id'],model=name,context=context,history_hash=history_hash,order=order,repeat=repeat,
                                 margin=measure['margin'],stance=measure['stance'],finite=finite,
                                 repeat_delta=measure['margin']-previous if previous is not None else None,
                                 dtype=str(next(runtime.model.parameters()).dtype) if hasattr(runtime,'model') else None,model_revision=getattr(runtime,'revision',None),tokenizer_revision=getattr(runtime,'tokenizer_revision',None),device=runtime.device,
                                 measurement=measure)
                        rows.append(row);previous=measure['margin'];progress(len(rows),total)
                        if not finite or any(n<1 for n in measure['token_counts']):
                            write(path/'audit/stance_audit.json',dict(protocol_hash=protocol_hash,status='failed',rows=rows,error='Invalid audit candidate likelihood/tokenization'))
                            raise ValueError('Invalid audit candidate likelihood/tokenization')
    summary=[]
    for name in runtimes:
        subset=[r for r in rows if r['model']==name]
        deltas=[abs(r['repeat_delta']) for r in subset if r['repeat_delta'] is not None]
        effects=[]
        for row in subset:
            if row['order']=='no_first' and row['repeat']==0:
                original=next(r for r in subset if r['question_id']==row['question_id'] and r['context']==row['context'] and r['order']=='yes_first' and r['repeat']==0)
                effects.append(row['margin']-original['margin'])
        summary.append(dict(model=name,probes=len(subset),max_repeat_margin_delta=max(deltas,default=0),mean_abs_order_margin_delta=average([abs(v) for v in effects]),
                            saturated_fraction=sum(abs(r['stance'])>.99 for r in subset)/len(subset),
                            mean_abs_margin=average([abs(r['margin']) for r in subset])))
    passed=all(r['max_repeat_margin_delta']<=tolerance for r in summary)
    cross_model=[]
    names=list(runtimes)
    if len(names)==2:
        for row in rows:
            if row['model']==names[0] and row['order']=='yes_first' and row['repeat']==0:
                peer=next(r for r in rows if r['model']==names[1] and r['history_hash']==row['history_hash'] and r['order']=='yes_first' and r['repeat']==0)
                cross_model.append(dict(question_id=row['question_id'],context=row['context'],history_hash=row['history_hash'],model_a=names[0],model_b=names[1],margin_a=row['margin'],margin_b=peer['margin'],margin_difference=peer['margin']-row['margin'],stance_difference=peer['stance']-row['stance']))
    csv_write(path/'audit/cross_model.csv',cross_model)
    result=dict(cross_model=cross_model,protocol_hash=protocol_hash,status='passed' if passed else 'failed',tolerance=tolerance,summary=summary,rows=rows,
                note='Identical raw histories, native model chat templates. Probe-order variants are diagnostics only; primary scoring remains unchanged. Score-scale differences are not calibrated away.')
    write(path/'audit/stance_audit.json',result)
    flat=[{k:v for k,v in r.items() if k!='measurement'} for r in rows];csv_write(path/'audit/stance_audit.csv',flat)
    csv_write(path/'audit/summary.csv',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(18,4),layout='constrained')
    for name in runtimes:
        subset=[r for r in rows if r['model']==name]
        axes[0].hist([r['margin'] for r in subset if r['repeat']==0 and r['order']=='yes_first'],bins=12,alpha=.5,label=name.split('/')[-1])
        axes[1].plot([abs(r['repeat_delta']) for r in subset if r['repeat_delta'] is not None],marker='.',label=name.split('/')[-1])
        primary=[r for r in subset if r['order']=='yes_first' and r['repeat']==0]
        reverse=[r for r in subset if r['order']=='no_first' and r['repeat']==0]
        axes[2].scatter([r['margin'] for r in primary],[r['margin'] for r in reverse],s=15,label=name.split('/')[-1])
    if cross_model:axes[3].scatter([r['margin_a'] for r in cross_model],[r['margin_b'] for r in cross_model],s=15,label='Identical histories')
    axes[3].set(xlabel=names[0].split('/')[-1] if names else 'Model A',ylabel=names[1].split('/')[-1] if len(names)>1 else 'Model B')
    for ax,title in zip(axes,['Identical-history margin distribution','Absolute repeat margin differences','Primary versus reversed presentation','Cross-model raw margins']):
        ax.set_title(title)
        if ax.has_data():ax.legend(fontsize=6)
    axes[0].set(xlabel='Raw margin',ylabel='Contexts');axes[1].set(xlabel='Context index',ylabel='Absolute delta');axes[2].set(xlabel='Yes first margin',ylabel='No first margin')
    for ext in ('png','pdf'):fig.savefig(path/f'audit/stance_audit.{ext}',dpi=150)
    plt.close(fig)
    if not passed:raise ValueError('Stance repeatability failed; inspect audit before collection')
    return result
