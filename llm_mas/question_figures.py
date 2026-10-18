"""Only per-question, single-repetition PNGs; raw analysis tables live elsewhere."""
from collections import defaultdict
from html import escape
import json
from pathlib import Path

from .output_state import measurement,source
from .storage import write
from .trajectories import conversations


def group_state(rows):
    if not rows:return 'Missing'
    if any(r.get('choice_status')!='valid' or r.get('answer') not in ('Yes','No') for r in rows):return 'Invalid'
    return rows[0]['answer'] if len({r['answer'] for r in rows})==1 else 'Split'


def render(path,output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    path=Path(path);folder=Path(output) if output else path/'figures';folder.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(lambda:defaultdict(dict))
    for file,record in conversations(path):
        if not record.get('dataset'):continue
        config=record['config']
        key=(record['question_id'],record.get('repetition',0),json.dumps(config['initialization']),config.get('decoding','greedy'))
        composition=tuple(config['models'] if len(config['models'])>1 else config['models']*config['agents'])
        schedule=config['schedule']
        if schedule in grouped[key][composition]:raise ValueError('Duplicate conversation coordinate; do not merge or average runs')
        grouped[key][composition][schedule]=(file.stem,record)
    entries=[]
    states=('Yes','No','Split','Invalid','Missing');colors=('#0072B2','#D55E00','#CC79A7','#888888','#eeeeee')
    for index,(key,compositions) in enumerate(sorted(grouped.items())):
        qid,repetition,initialization,decoding=key
        schedules=sorted({s for branches in compositions.values() for s in branches if s!='no_peer'}) or ['synchronous']
        for schedule in schedules:
            fig,axes=plt.subplots(len(compositions),5,figsize=(24,3.4*len(compositions)),squeeze=False,layout='constrained')
            source_rows=[]
            def order(item):
                code=''.join('Q' if 'Qwen' in m else 'S' if 'SmolLM2' in m else '?' for m in item[0])
                return ({'QQ':0,'SS':1,'QS':2,'SQ':3}.get(code,4),item[0])
            for i,(composition,branches) in enumerate(sorted(compositions.items(),key=order)):
                code=''.join('Q' if 'Qwen' in m else 'S' if 'SmolLM2' in m else '?' for m in composition)
                label=code if '?' not in code else '/'.join(m.split('/')[-1] for m in composition)
                for j,condition in enumerate((schedule,'no_peer')):
                    pair=branches.get(condition)
                    if pair:
                        task,record=pair;rows=record['rows'];agents=list(dict.fromkeys(r['agent'] for r in rows))
                        for agent in agents:
                            own=sorted([r for r in rows if r['agent']==agent],key=lambda r:r['round'])
                            for col,values in ((2*j,[measurement(r).get('stance',float('nan')) for r in own]),(2*j+1,[r.get('sentiment',float('nan')) for r in own])):
                                axes[i,col].plot([r['round'] for r in own],values,marker='o',markersize=3,label=agent)
                        mode='Decision-time stance' if all(r.get('decision_time') for r in rows) else 'Post-response probe stance (legacy)'
                        axes[i,2*j].set_ylabel(mode);axes[i,2*j+1].set_ylabel('Public-message sentiment')
                        source_rows.append(dict(composition=list(composition),schedule=condition,conversation=task,pair_id=record.get('pair_id'),stance_sources=sorted({source(r) for r in rows})))
                    else:
                        task='not recorded'
                        for col in (2*j,2*j+1):axes[i,col].text(.5,.5,'Condition not recorded',ha='center',transform=axes[i,col].transAxes)
                    for col,kind in ((2*j,'Stance'),(2*j+1,'Sentiment')):
                        ax=axes[i,col];ax.set_title(f'{label}\n{condition} · {task} · {kind}',fontsize=8);ax.set_xlabel('Round');ax.set_ylim(-1.05,1.05);ax.axhline(0,color='gray',lw=.6)
                        if pair:ax.legend(fontsize=7)
                rounds=sorted({r['round'] for condition in (schedule,'no_peer') if condition in branches for r in branches[condition][1]['rows']})
                matrix=[]
                for condition in (schedule,'no_peer'):
                    rows=branches[condition][1]['rows'] if condition in branches else []
                    matrix.append([states.index(group_state([r for r in rows if r['round']==t])) for t in rounds])
                ax=axes[i,4];ax.imshow(matrix,cmap=ListedColormap(colors),norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5,4.5],5),aspect='auto')
                ax.set_xticks(range(len(rounds)),rounds);ax.set_yticks([0,1],[schedule.replace('_',' '),'no peer'],fontsize=8);ax.set_title('Explicit-answer group states',fontsize=9);ax.set_xlabel('Round')
                for row in range(2):
                    for col in range(len(rounds)):
                        value=states[matrix[row][col]];ax.text(col,row,value,ha='center',va='center',fontsize=7,color='black' if value in ('Split','Missing') else 'white')
            record=next(iter(next(iter(compositions.values())).values()))[1]
            fig.suptitle(f'{record["scenario"].get("question",qid)}\nRun {repetition+1} (saved repetition {repetition}) · exact scores, no averaging · Q: Qwen2.5, S: SmolLM2 · initialization {initialization} · {decoding}',fontsize=11)
            filename=f'question_{index:03d}_{qid}_rep_{repetition}_{schedule}.png'
            fig.savefig(folder/filename,dpi=150);plt.close(fig)
            entries.append(dict(question_id=qid,repetition=repetition,initialization=initialization,decoding=decoding,peer_schedule=schedule,png=filename,conversations=source_rows))
    # Provenance is analysis data, not a figure-side CSV/PDF export.
    write((folder.parent if output else path)/'analysis/question_figures.json',dict(profile='question_png',figures=entries,note='Each curve is one agent in one saved conversation. Decision-time scores are used when recorded; historical rows retain the separately probed post-response scores. Sentiment is measured on the completed public message.'))
    return entries


def report(path,data=None,output=None):
    path=Path(path);destination=Path(output) if output else path;destination.mkdir(parents=True,exist_ok=True)
    entries=render(path,output=destination/'figures' if output else None)
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Question figures</title><style>body{font:16px system-ui;max-width:1600px;margin:2rem auto;padding:1rem}img{max-width:100%;height:auto}a{color:#005ea8}</style><h1>Question figures</h1>',
        '<p>Single-repetition stance, public-message sentiment, and explicit-answer group states. No averaging. Sampling can select an answer whose raw score is lower. Missing means no recorded observation; Invalid means an unparseable recorded answer. Historical experiments retain their original probe measurements.</p>']
    for entry in entries:body.append(f'<section><h2>{escape(entry["question_id"])} · run {entry["repetition"]+1}</h2><img loading="lazy" src="figures/{escape(entry["png"])}" alt="Stance, sentiment and group states for one question and repetition"></section>')
    body.append('<p>Analysis, audit, conversations, initializations, inputs and tables remain saved for further analysis.</p>')
    (destination/'index.html').write_text('\n'.join(body))
    write(destination/'analysis/report_manifest.json',dict(version='question-png-v1',figures=entries))
    return entries
