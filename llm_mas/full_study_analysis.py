"""Variable-horizon figures and tables from saved conversations only."""
import csv
import json
import zipfile
from pathlib import Path
from collections import defaultdict


def table(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def export(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=Path(path);figdir=path/'figures';tabdir=path/'tables'
    figdir.mkdir(exist_ok=True);tabdir.mkdir(exist_ok=True)
    outcomes=[];trajectory=[];stopping=[];files=[]
    def save(fig,name):
        for ext in ('png','pdf'):
            target=figdir/f'{name}.{ext}';fig.savefig(target,bbox_inches='tight');files.append(target)
        plt.close(fig)
    for file in sorted((path/'conversations').glob('*.json')):
        d=json.loads(file.read_text());rows=d['rows'];c=d['config'];names=sorted({r['agent'] for r in rows});end=max(r['round'] for r in rows)
        final=[next(r for r in reversed(rows) if r['agent']==n) for n in names]
        invalid=any(r['strategy'] is None for r in final)
        agreement=not invalid and len({r['strategy'] for r in final})==1
        reason='fixed_horizon' if c.get('observe_stopping') else (d.get('termination') or {}).get('reason','fixed_horizon')
        triggers=[r['round'] for r in rows if (r.get('termination') or {}).get('reason')=='system_detected_stability']
        first=min(triggers) if triggers else None
        later=False
        if first is not None:
            at={r['agent']:r['strategy'] for r in rows if r['round']==first}
            later=any(r['round']>first and r['strategy']!=at[r['agent']] for r in rows)
        outcomes.append(dict(task=file.stem,scenario=d['scenario']['id'],models=' / '.join(c['models']),persona=c.get('persona',False),
            initialization=json.dumps(c['initialization']),schedule=c['schedule'],decision_mode=c.get('decision_mode','scored'),
            final_round=end,responses=len(rows),reason=reason,outcome='invalid' if invalid else f'agree_{final[0]["strategy"]}' if agreement else 'disagree',
            invalid_responses=sum(r['strategy'] is None for r in rows),first_stop_round=first,
            later_revision=later if c.get('observe_stopping') and first is not None else None,
            potential_rounds_saved=end-first if c.get('observe_stopping') and first is not None else None,
            generated_tokens=sum((r.get('generation') or {}).get('generated_tokens',0) for r in rows),
            potential_tokens_saved=sum((r.get('generation') or {}).get('generated_tokens',0) for r in rows if first is not None and r['round']>first) if c.get('observe_stopping') and first is not None else None))
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        seen={n:set() for n in names}
        for i,n in enumerate(names):
            agent=[r for r in rows if r['agent']==n];x=[r['round'] for r in agent]
            axes[0,0].plot(x,[r['post']['stance'] for r in agent],marker='o',label=n)
            axes[0,1].plot(x,[r.get('sentiment',float('nan')) for r in agent],marker='o',label=n)
            axes[1,0].step(x,[r['strategy'] if r['strategy'] is not None else float('nan') for r in agent],where='post',label=n)
            axes[1,1].plot(x,[r['post']['margin'] for r in agent],marker='.',label=n)
            previous=None
            for r in agent:
                term=r.get('termination') or {};sim=term.get('similarities',{}).get(n)
                repeat=r['message'] in seen[n];seen[n].add(r['message'])
                trajectory.append(dict(task=file.stem,agent=n,round=r['round'],strategy=r['strategy'],margin=r['post']['margin'],stance=r['post']['stance'],
                    sentiment=r.get('sentiment'),exact_repeat=repeat,adjacent_message_similarity=sim,explanation_similarity=r.get('explanation_similarity'),adjacent_explanation_similarity=r.get('adjacent_explanation_similarity'),
                    switched=previous is not None and previous['strategy'] is not None and r['strategy'] is not None and r['strategy']!=previous['strategy']))
                previous=r
        for ax,title in zip(axes.flat,['Post-response stance','Sentiment','Public choice (gaps = invalid)','Raw post-response margin']):
            ax.set(title=title,xlabel='Round');ax.legend()
            if first is not None:ax.axvline(first,color='gray',linestyle=':',label='First stopping trigger')
        axes[0,0].set_ylim(-1.05,1.05);axes[0,1].set_ylim(-1.05,1.05);axes[1,0].set_yticks([1,2])
        fig.suptitle(f'{file.stem} | {d["scenario"]["title"]}\n{c.get("decision_mode","scored")} | {c["schedule"]} | {reason}')
        save(fig,file.stem+'_trajectory')
        for r in rows:
            if r['agent']==names[0] and r.get('termination'):
                stopping.append(dict(task=file.stem,**r['termination']))
    table(tabdir/'outcomes.csv',outcomes);table(tabdir/'trajectories.csv',trajectory)
    (tabdir/'stopping_checks.json').write_text(json.dumps(stopping,indent=2))
    groups=defaultdict(list)
    for r in outcomes:groups[(r['scenario'],r['decision_mode'],r['schedule'],r['persona'])].append(r)
    summaries=[]
    for key,group in groups.items():
        summaries.append(dict(scenario=key[0],decision_mode=key[1],schedule=key[2],persona=key[3],conversations=len(group),
            agree_1=sum(r['outcome']=='agree_1' for r in group),agree_2=sum(r['outcome']=='agree_2' for r in group),
            disagree=sum(r['outcome']=='disagree' for r in group),invalid=sum(r['outcome']=='invalid' for r in group),
            mean_final_round=sum(r['final_round'] for r in group)/len(group)))
    table(tabdir/'condition_summary.csv',summaries)
    fig,ax=plt.subplots(figsize=(14,max(4,len(summaries)*.23)),layout='constrained');left=[0]*len(summaries)
    for label in ['agree_1','agree_2','disagree','invalid']:
        values=[r[label] for r in summaries];ax.barh(range(len(summaries)),values,left=left,label=label);left=[a+b for a,b in zip(left,values)]
    ax.set_yticks(range(len(summaries)),[f'{r["scenario"]} / {r["decision_mode"]} / {r["schedule"]} / persona={r["persona"]}' for r in summaries]);ax.set_xlabel('Conversations at their actual endpoint');ax.legend();save(fig,'outcomes')
    fig,ax=plt.subplots(layout='constrained');reasons=sorted({r['reason'] for r in outcomes})
    for reason in reasons:ax.hist([r['final_round'] for r in outcomes if r['reason']==reason],bins=range(max(r['final_round'] for r in outcomes)+2),alpha=.6,label=reason)
    ax.set(xlabel='Final round',ylabel='Conversations');ax.legend();save(fig,'lengths')
    rounds=[]
    for t in sorted({r['round'] for r in trajectory}):
        group=[r for r in trajectory if r['round']==t];sims=[r['adjacent_message_similarity'] for r in group if r['adjacent_message_similarity'] is not None]
        explanations=[r['explanation_similarity'] for r in group if r['explanation_similarity'] is not None]
        rounds.append(dict(mean_explanation_similarity=sum(explanations)/len(explanations) if explanations else None,round=t,observed_conversations=len({r['task'] for r in group}),switches=sum(r['switched'] for r in group),exact_repeat_fraction=sum(r['exact_repeat'] for r in group)/len(group),mean_similarity=sum(sims)/len(sims) if sims else None))
    table(tabdir/'round_summary.csv',rounds)
    fig,axes=plt.subplots(4,1,figsize=(9,10),layout='constrained')
    for ax,key in zip(axes,['observed_conversations','switches','mean_similarity','mean_explanation_similarity']):
        ax.plot([r['round'] for r in rounds],[r[key] if r[key] is not None else float('nan') for r in rounds],marker='o');ax.set(xlabel='Round',ylabel=key)
    save(fig,'round_summary')
    html='<h1>Full behavioral study</h1><p>Actual observed horizons; no padding. Dotted trajectory lines mark the first stopping trigger.</p>'
    html+=''.join(f'<p><a href="{f.name}">{f.stem}</a></p>' for f in files if f.suffix=='.png')
    (figdir/'index.html').write_text(html)
    with zipfile.ZipFile(path/'paper_outputs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in [figdir,tabdir]:
            for f in folder.iterdir():z.write(f,str(f.relative_to(path)))
