"""Experiment D behavioral exports. No model loading or internal interventions."""
import csv
import json
import re
import zipfile
from collections import defaultdict
from html import escape
from pathlib import Path
from .experiment_c import export as trajectories, summary


def table(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def export(path):
    path=Path(path)
    trajectories(path, persona_comparison=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    records={f.stem:json.loads(f.read_text()) for f in sorted((path/'conversations').glob('*.json'))}
    groups=defaultdict(list); responses=[]; outcomes=[]; changes=[]
    for task, rec in records.items():
        c=rec['config'];scenario=rec['scenario']['id']
        kind='homogeneous' if len(set(c['models']))==1 else 'heterogeneous'
        groups[(scenario,kind)].append((task,rec))
        base=dict(task=task,scenario=scenario,kind=kind,persona=bool(c.get('persona')),schedule=c['schedule'],initialization=json.dumps(c['initialization']))
        outcomes.append(dict(summary(task,rec),persona=bool(c.get('persona'))))
        for agent in sorted({r['agent'] for r in rec['rows']}):
            seq=sorted((r for r in rec['rows'] if r['agent']==agent),key=lambda r:r['round'])
            seen=set();previous=None
            for r in seq:
                decision=(r.get('decision') or {}).get('margin');post=r['post']['margin'];before=r['before']['margin']
                explanation=r.get('argument','')
                # Observable formatting checks, not a semantic compliance verdict.
                headings=bool(re.search(r'(?im)^\s*(?:arguments?|explanation)\s*:',explanation))
                bullets=bool(re.search(r'(?m)^\s*(?:[-*•]|\d+[.)])\s+',explanation))
                paragraphs=len([s for s in re.split(r'\n\s*\n',explanation.strip()) if s.strip()])
                item=dict(base,agent=agent,model=r['model'],round=r['round'],strategy=r['strategy'],
                    forced=r.get('forced',False),before_margin=before,decision_margin=decision,post_margin=post,
                    post_stance=r['post']['stance'],sentiment=r.get('sentiment'),
                    delivery_margin_change=decision-before if decision is not None and r.get('deliveries') else None,
                    own_response_margin_change=post-decision if decision is not None else None,
                    public_post_mismatch=(post>=0)!=(r['strategy']==1),
                    repeated_message=r['message'] in seen,has_heading=headings,has_bullets=bullets,
                    paragraph_count=paragraphs,paragraph_format_pass=paragraphs==1 and not headings and not bullets,
                    public_switch=previous is not None and previous['strategy']!=r['strategy'],
                    scored_switch=previous is not None and previous.get('decision') is not None and decision is not None and previous['strategy']!=r['strategy'],
                    forced_initial_transition=previous is not None and previous.get('forced',False) and r['round']==1)
                responses.append(item)
                if previous:
                    changes.append(dict(base,agent=agent,round=r['round'],stance_change=r['post']['stance']-previous['post']['stance'],
                        sentiment_change=r['sentiment']-previous['sentiment'] if r.get('sentiment') is not None and previous.get('sentiment') is not None else None))
                seen.add(r['message']);previous=r
    table(path/'tables/behavioral_responses.csv',responses)
    table(path/'tables/stance_sentiment_changes.csv',changes)
    # Compare matched individual responses; preserve scenario, checkpoint and start.
    matched=[]
    for factor in ('persona','schedule'):
        lookup={}
        for row in responses:
            key=tuple(row[k] for k in ('scenario','agent','model','round','initialization'))
            config=records[row['task']]['config']
            key+=(tuple(config['models']),row['schedule'] if factor=='persona' else row['persona'])
            lookup.setdefault(key,{})[row[factor]]=row
        left,right=(False,True) if factor=='persona' else ('synchronous','sequential')
        for pair in lookup.values():
            if left not in pair or right not in pair:continue
            a,b=pair[left],pair[right]
            matched.append(dict(factor=factor,task_left=a['task'],task_right=b['task'],agent=a['agent'],round=a['round'],
                same_public_choice=a['strategy']==b['strategy'],post_stance_right_minus_left=b['post_stance']-a['post_stance'],
                sentiment_right_minus_left=b['sentiment']-a['sentiment'] if a['sentiment'] is not None and b['sentiment'] is not None else None))
    table(path/'tables/matched_comparisons.csv',matched)
    for (scenario,kind),items in sorted(groups.items()):
        items.sort(key=lambda x:(tuple(x[1]['config']['models']),str(x[1]['config']['initialization']),x[1]['config'].get('persona',False),x[1]['config']['schedule']))
        labels=[];values=[];rounds=max(r['round'] for _,rec in items for r in rec['rows'])+1
        for task,rec in items:
            c=rec['config']
            for agent in sorted({r['agent'] for r in rec['rows']}):
                seq=sorted((r for r in rec['rows'] if r['agent']==agent),key=lambda r:r['round'])
                if len(seq)!=rounds:raise ValueError('Incomplete Experiment D trajectory')
                labels.append(f"{task} {seq[0]['model'].split('/')[-1]} · {agent[-1]} · {'persona' if c.get('persona') else 'no persona'} · {'natural' if c['initialization']=='natural' else 'forced'} · {c['schedule']}")
                values.append([r['strategy'] for r in seq])
        fig,ax=plt.subplots(figsize=(15,max(5,len(labels)*.25)),layout='constrained')
        img=ax.imshow(values,aspect='auto',interpolation='nearest',cmap=ListedColormap(['#2474b5','#d48c27']),vmin=.5,vmax=2.5)
        ax.set_yticks(range(len(labels)),labels,fontsize=7);ax.set_xticks(range(rounds))
        ax.set(xlabel='Round (0 = initialization)',title=f'{scenario} · {kind} · public choices')
        bar=fig.colorbar(img,ax=ax,ticks=[1,2]);bar.ax.set_yticklabels(['Strategy 1','Strategy 2'])
        fig.savefig(path/'figures'/f'{scenario}_{kind}_public_choices.png',dpi=160);plt.close(fig)
    figure_names=sorted(f.name for f in (path/'figures').glob('*.png'))
    coverage=dict(profile='experiment_d',conversations=len(records),responses=len(responses),figures=figure_names,
        analysis='Behavioral only; no all-layer replay or interventions',format_checks='Heuristic headings, bullets and paragraph count; not semantic validation')
    (path/'figures/coverage.json').write_text(json.dumps(coverage,indent=2))
    (path/'figures/index.html').write_text('<!doctype html><meta charset="utf-8"><title>Experiment D</title><h1>Experiment D</h1><p>Behavioral results. Agreement is limited to the observed horizon.</p>'+''.join(f'<p><a href="../tables/{escape(f.name)}">{escape(f.name)}</a></p>' for f in sorted((path/'tables').glob('*.csv')))+''.join(f'<h2>{escape(n)}</h2><img style="max-width:100%" src="{escape(n)}">' for n in figure_names))
    with zipfile.ZipFile(path/'paper_outputs.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('figures','tables'):
            for f in sorted((path/folder).glob('*')):
                if f.is_file():archive.write(f,f.relative_to(path))
