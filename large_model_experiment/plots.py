"""Per-question PNGs without stance: sentiment curves, per-agent answer states and group states.

Same layout logic as llm_mas/question_figures.py, with the stance panels replaced by
per-agent explicit-answer states (API models expose no answer-token logits).
"""
from collections import defaultdict
from html import escape
import json
from pathlib import Path

from llm_mas.question_figures import group_state
from llm_mas.storage import write
from llm_mas.trajectories import conversations

STATES = ('Yes', 'No', 'Split', 'Invalid', 'Missing')
COLORS = ('#0072B2', '#D55E00', '#CC79A7', '#888888', '#eeeeee')


def code(models, labels):
    return ''.join(labels.get(m, '?') for m in models)


def render(path, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    path = Path(path); folder = path/'figures'; folder.mkdir(parents=True, exist_ok=True)
    cmap, norm = ListedColormap(COLORS), BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5, 4.5], 5)
    grouped = defaultdict(lambda: defaultdict(dict))
    for file, record in conversations(path):
        config = record['config']
        grouped[(record['question_id'], record.get('repetition', 0))][tuple(config['models'])][config['schedule']] = (file.stem, record)
    order = {c: i for i, c in enumerate(('CC', 'OO', 'CO', 'OC'))}
    entries = []
    for index, ((qid, repetition), compositions) in enumerate(sorted(grouped.items())):
        items = sorted(compositions.items(), key=lambda item: (order.get(code(item[0], labels), 9), item[0]))
        fig, axes = plt.subplots(len(items), 4, figsize=(22, 3.4*len(items)), squeeze=False, layout='constrained',
                                 gridspec_kw=dict(width_ratios=[1, 1, 1.2, 1.2]))
        sources = []
        for i, (composition, branches) in enumerate(items):
            label = code(composition, labels)
            rounds = sorted({r['round'] for b in branches.values() for r in b[1]['rows']})
            agent_matrix, agent_labels, group_matrix = [], [], []
            for j, condition in enumerate(('synchronous', 'no_peer')):
                ax = axes[i, j]
                pair = branches.get(condition)
                rows = pair[1]['rows'] if pair else []
                if pair:
                    sources.append(dict(composition=list(composition), schedule=condition, conversation=pair[0], pair_id=pair[1].get('pair_id')))
                    for agent in dict.fromkeys(r['agent'] for r in rows):
                        own = sorted([r for r in rows if r['agent'] == agent], key=lambda r: r['round'])
                        model = labels.get(own[0]['model'], '?')
                        ax.plot([r['round'] for r in own], [r.get('sentiment', float('nan')) for r in own], marker='o', markersize=3, label=f'{agent} ({model})')
                        agent_labels.append(f'{agent} ({model}) · {"peer" if condition == "synchronous" else "no peer"}')
                        by_round = {r['round']: r for r in own}
                        agent_matrix.append([STATES.index(by_round[t]['answer']) if t in by_round and by_round[t].get('answer') in ('Yes', 'No')
                                             else STATES.index('Invalid' if t in by_round else 'Missing') for t in rounds])
                    ax.legend(fontsize=7)
                else:
                    ax.text(.5, .5, 'Condition not recorded', ha='center', transform=ax.transAxes)
                group_matrix.append([STATES.index(group_state([r for r in rows if r['round'] == t])) for t in rounds])
                ax.set_title(f'{label}\n{condition} · {pair[0] if pair else "not recorded"} · Sentiment', fontsize=8)
                ax.set_xlabel('Round'); ax.set_ylabel('Public-message sentiment'); ax.set_ylim(-1.05, 1.05); ax.axhline(0, color='gray', lw=.6)
            for col, matrix, ylabels, title in ((2, agent_matrix, agent_labels, 'Explicit answer per agent'),
                                                (3, group_matrix, ['synchronous', 'no peer'], 'Explicit-answer group states')):
                ax = axes[i, col]
                ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
                ax.set_xticks(range(len(rounds)), rounds); ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
                ax.set_title(f'{label} · {title}', fontsize=9); ax.set_xlabel('Round')
                for r, values in enumerate(matrix):
                    for c, value in enumerate(values):
                        ax.text(c, r, STATES[value], ha='center', va='center', fontsize=7, color='black' if STATES[value] in ('Split', 'Missing') else 'white')
        record = next(iter(next(iter(compositions.values())).values()))[1]
        legend = ' · '.join(f'{v}: {k}' for k, v in labels.items())
        reference = record.get('evaluation', {}).get('reference_answer')
        fig.suptitle(f'{record["scenario"].get("question", qid)} (label: {reference})\nRun {repetition+1} · no averaging · {legend} · no stance (API models expose no answer-token logits)', fontsize=11)
        filename = f'question_{index:03d}_{qid}_rep_{repetition}_synchronous.png'
        fig.savefig(folder/filename, dpi=150); plt.close(fig)
        entries.append(dict(question_id=qid, repetition=repetition, peer_schedule='synchronous', png=filename, conversations=sources))
    write(path/'analysis/question_figures.json', dict(profile='question_png_no_stance', figures=entries,
          note='Sentiment is measured on the completed public message. Answer states are the explicit answers each agent wrote.'))
    return entries


def report(path, labels, data=None):
    path = Path(path)
    entries = render(path, labels)
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Question figures</title><style>body{font:16px system-ui;max-width:1600px;margin:2rem auto;padding:1rem}img{max-width:100%;height:auto}</style><h1>Question figures</h1>',
            '<p>Single-repetition public-message sentiment, per-agent explicit answers, and explicit-answer group states. No stance: API models expose no answer-token logits. Invalid means no parseable answer line.</p>']
    for entry in entries:
        body.append(f'<section><h2>{escape(entry["question_id"])} · run {entry["repetition"]+1}</h2><img loading="lazy" src="figures/{escape(entry["png"])}" alt="Sentiment, answers and group states for one question"></section>')
    (path/'index.html').write_text('\n'.join(body))
    write(path/'analysis/report_manifest.json', dict(version='question-png-no-stance-v1', figures=entries))
    return entries
