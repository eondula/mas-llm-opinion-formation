"""All plotting reads saved data; never loads language models."""
import csv
import json
from pathlib import Path

def render(path):
    path = Path(path)
    if (path/'config.json').exists() and json.loads((path/'config.json').read_text()).get('export_profile') == 'experiment_c':
        from .experiment_c import export
        return export(path)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    path = Path(path)
    plt.rcParams.update({'font.size':12,'axes.labelweight':'bold','axes.titleweight':'bold','lines.linewidth':2.3,'savefig.bbox':'tight'})
    folder = path/'figures'
    folder.mkdir(exist_ok=True)
    files, records = [], []
    for source in sorted((path/'conversations').glob('*.json')):
        data = json.loads(source.read_text())
        rows = data['rows']
        agents = list(dict.fromkeys(r['agent'] for r in rows))
        title = f"{data['scenario']['title']} | {data['config']['schedule']} | {data['config']['initialization']} | {data['config']['argument_mode']}"
        fig, axes = plt.subplots(1,2,figsize=(12,4), layout='constrained')
        deltafig, dax = plt.subplots(len(agents),2,figsize=(11,3*len(agents)),squeeze=False,layout='constrained')
        transcript = [title, 'Synchronous: each response uses prior-round peer messages. Sequential: later speakers see current-round messages.','']
        for index, agent in enumerate(agents):
            subset = [r for r in rows if r['agent']==agent]
            rounds = [r['round'] for r in subset]
            stance = [r['post']['stance'] for r in subset]
            sentiment = [r.get('sentiment',float('nan')) for r in subset]
            axes[0].plot(rounds,stance,marker='o',markersize=4,label=agent)
            axes[1].plot(rounds,sentiment,marker='o',markersize=4,label=agent)
            ds, dv = np.abs(np.diff(stance)), np.abs(np.diff(sentiment))
            dax[index,0].plot(rounds[1:],ds,label='Absolute stance change')
            dax[index,0].plot(rounds[1:],dv,label='Absolute sentiment change')
            dax[index,0].set(title=agent,xlabel='Round',ylabel='Absolute change')
            dax[index,0].legend()
            dax[index,1].scatter(ds,dv)
            for t,x,y in zip(rounds[1:],ds,dv):
                if np.isfinite(y): dax[index,1].annotate(str(t),(x,y))
            dax[index,1].set(xlabel='Absolute stance change',ylabel='Absolute sentiment change')
            for j,r in enumerate(subset):
                records.append(dict(task=source.stem,agent=agent,round=r['round'],strategy=r['strategy'],post_stance=r['post']['stance'],
                    decision_stance=r['decision']['stance'] if r['decision'] else None,margin=r['post']['margin'],sentiment=r.get('sentiment'),
                    absolute_stance_change=float(ds[j-1]) if j else None,absolute_sentiment_change=float(dv[j-1]) if j and np.isfinite(dv[j-1]) else None))
        for r in rows:
            transcript.extend([f"Round {r['round']} — {r['agent']} ({r['model']}) — S{r['strategy']} — post stance {r['post']['stance']:+.6f}",r['message'],''])
        for ax,label in zip(axes,('Post-response stance: S2 (−) to S1 (+)','Public-message sentiment')):
            ax.set(xlabel='Round (0 = initialization)',ylabel=label,ylim=(-1.05,1.05))
            ax.axhline(0,color='gray',linewidth=.7)
            ax.legend()
        fig.suptitle(title,fontsize=11)
        if not any('sentiment' in r for r in rows): axes[1].text(.5,.5,'Sentiment not measured',transform=axes[1].transAxes,ha='center')
        for f,stem in [(fig,source.stem+'_stance_sentiment'),(deltafig,source.stem+'_absolute_changes')]:
            from .plot_data import export as export_plot_data
            export_plot_data(f,folder/(stem+'.csv'))
            for ext in ('png','pdf'):
                filename=f'{stem}.{ext}'; f.savefig(folder/filename,dpi=180); files.append(filename)
            plt.close(f)
        (folder/f'{source.stem}_transcript.txt').write_text('\n'.join(transcript))
    if records:
        with (folder/'trajectories.csv').open('w') as out:
            writer=csv.DictWriter(out,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    research_files = []
    for source in sorted((path/'measurements').glob('*.json')):
        if source.stem not in ('validation','history'): continue
        data = json.loads(source.read_text())
        fig, ax = plt.subplots(figsize=(10,5), layout='constrained')
        if source.stem == 'validation':
            for i,r in enumerate(data):
                ax.plot([0,1,2,3],[r['baseline']['margin'],r['repeat']['margin'],r['swapped_semantic_margin'],r['reversed_presentation']['margin']],marker='o',label=r['model']+' / '+r['scenario'])
            ax.set_xticks([0,1,2,3],['Original','Repeat','Candidate swap','Presentation reversal'])
        elif source.stem == 'history':
            groups = {}
            for r in data: groups.setdefault((r['model'],r['scenario']),[]).append(r)
            for key,group in groups.items(): ax.plot(range(len(group)),[r['delta'] for r in group],marker='o',label=' / '.join(key))
            ax.set_xticks([0,1,2,3],['Self 1 / peer 1','Self 1 / peer 2','Self 2 / peer 1','Self 2 / peer 2'])
        ax.set_ylabel('Raw margin' if source.stem=='validation' else 'Peer-associated margin change')
        ax.axhline(0,color='gray',linewidth=.8); ax.legend(fontsize=8)
        from .plot_data import export as export_plot_data
        export_plot_data(fig,folder/(source.stem+'.csv'))
        for ext in ('pdf','png'): fig.savefig(folder/(source.stem+'.'+ext),dpi=180)
        research_files.append(source.stem)
        plt.close(fig)
    coverage = {'conversation_figures':files,'research_figures':research_files,'pending':['measurement_validation','controlled_history_effects','held_out_prediction','mechanistic_interventions','group_summary'],
                'note':'Conversation exports are available. This is not a complete paper figure set.'}
    (folder/'coverage.json').write_text(json.dumps(coverage,indent=2))

    from .reporting import export
    export(path)
