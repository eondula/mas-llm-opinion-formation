"""Experiment C: grouped PNG figures and one outcome CSV per agent configuration type."""
import csv
import json
import zipfile
from collections import defaultdict
from html import escape
from pathlib import Path


def summary(task, record):
    c=record['config'];rows=record['rows'];agents=sorted({r['agent'] for r in rows})
    by_agent={a:sorted((r for r in rows if r['agent']==a),key=lambda r:r['round']) for a in agents}
    rounds=sorted({r['round'] for r in rows})
    states=[tuple(next(r['strategy'] for r in by_agent[a] if r['round']==t) for a in agents) for t in rounds]
    agreed=[len(set(s))==1 for s in states]
    stable=next((rounds[i] for i in range(len(states)) if agreed[i] and all(s==states[i] for s in states[i:])),None)
    result=dict(task=task,scenario=record['scenario']['id'],schedule=c['schedule'],argument_mode=c['argument_mode'],
                initialization=json.dumps(c['initialization']),update_rounds=c['rounds'],
                final_outcome='agreement' if agreed[-1] else 'disagreement',
                final_strategy=states[-1][0] if agreed[-1] else '',
                first_agreement_round=next((rounds[i] for i,v in enumerate(agreed) if v),None),
                stable_agreement_round=stable)
    for a in agents:
        seq=by_agent[a];last=seq[-1];key=a.lower().replace(' ','_')
        scored=[r for r in seq if r['decision'] is not None]
        result.update({key+'_model':last['model'],key+'_initial_strategy':seq[0]['strategy'],key+'_final_strategy':last['strategy'],
            key+'_decision_switches':sum(x['strategy']!=y['strategy'] for x,y in zip(scored,scored[1:])),
            key+'_final_decision_margin':last['decision']['margin'] if last['decision'] else None,
            key+'_final_post_margin':last['post']['margin'],key+'_final_post_stance':last['post']['stance'],
            key+'_final_sentiment':last.get('sentiment'),key+'_repeated_messages':len(seq)-len({r['message'] for r in seq})})
    return result


def export(path, persona_comparison=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=Path(path);folder=path/'figures';folder.mkdir(exist_ok=True)
    tables=path/'tables';tables.mkdir(exist_ok=True)
    groups=defaultdict(list);summaries={'homogeneous':[],'heterogeneous':[]}
    for f in sorted((path/'conversations').glob('*.json')):
        record=json.loads(f.read_text());c=record['config']
        kind='homogeneous' if len(set(c['models']))==1 else 'heterogeneous'
        groups[(record['scenario']['id'],kind,c['schedule'],'natural' if c['initialization']=='natural' else 'forced_A1_B2', ('persona' if c.get('persona') else 'no_persona') if persona_comparison else '')].append((f.stem,record))
        result=summary(f.stem,record)
        if persona_comparison:result['persona']=bool(c.get('persona'))
        summaries[kind].append(result)
    generated=[]
    for (scenario,kind,schedule,start,persona),records in groups.items():
        fig,axes=plt.subplots(len(records),2,figsize=(13,3.5*len(records)),squeeze=False,layout='constrained')
        for axrow,(task,rec) in zip(axes,records):
            agents=sorted({r['agent'] for r in rec['rows']})
            for i,a in enumerate(agents):
                seq=sorted((r for r in rec['rows'] if r['agent']==a),key=lambda r:r['round'])
                label=f"{a}: {seq[0]['model'].split('/')[-1]}"
                for ax,values in zip(axrow,([r['post']['stance'] for r in seq],[r.get('sentiment',float('nan')) for r in seq])):
                    ax.plot([r['round'] for r in seq],values,label=label,color=['#2474b5','#b52b85'][i%2],linestyle=['-','--'][i%2],marker=['o','s'][i%2],markersize=3)
            for ax,ylabel in zip(axrow,['Post-response stance (S2 − / S1 +)','Public-message sentiment']):
                ax.set(xlabel='Round (0 = initialization)',ylabel=ylabel,ylim=(-1.05,1.05))
                ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.15);ax.legend(fontsize=8)
            model=rec['rows'][0]['model'].split('/')[-1] if kind=='homogeneous' else 'Fixed heterogeneous pair'
            axrow[0].set_title(model+' · stance');axrow[1].set_title(model+' · sentiment')
        fig.suptitle(f"{records[0][1]['scenario']['title']}\n{kind.capitalize()} · {'Synchronous' if schedule=='synchronous' else 'Sequential (A then B)'} · {start.replace('_',' ')} · combined flexible reasoning")
        if persona_comparison:fig.suptitle(fig._suptitle.get_text()+' · '+persona.replace('_',' '))
        name=f'{scenario}_{kind}_{schedule}_{start}'+('_'+persona if persona else '')+'.png';fig.savefig(folder/name,dpi=180);plt.close(fig);generated.append(name)
    for kind,data in summaries.items():
        if not data:continue
        with (tables/f'{kind}_summary.csv').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Experiment C</title><h1>Experiment C</h1><p>Post-response stance and public-message sentiment. Stable agreement means the same joint choice persists through the observed final round; it is not a claim about later rounds.</p>'+''.join(f'<p><a href="../tables/{k}_summary.csv">{k.capitalize()} summary CSV</a></p>' for k in summaries)+''.join(f'<h2>{escape(n)}</h2><img style="max-width:100%" src="{escape(n)}">' for n in generated))
    (folder/'coverage.json').write_text(json.dumps(dict(profile='experiment_c',conversation_figures=generated,formats=['png'],summary_tables=[f'{k}_summary.csv' for k in summaries],note='Decision switches exclude assigned initial choices. Stable agreement is within the observed horizon.'),indent=2))
    with zipfile.ZipFile(path/'paper_outputs.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(list(folder.glob('*'))+list(tables.glob('*'))):
            if f.is_file():archive.write(f,f.relative_to(path))
