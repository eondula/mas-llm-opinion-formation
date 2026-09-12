"""Case-based analysis of saved Experiment C conversations; no new dialogue."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from .runner import write


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def agents(record):
    return sorted({r['agent'] for r in record['rows']})


def sequence(record, agent):
    return sorted((r for r in record['rows'] if r['agent'] == agent), key=lambda r: r['round'])


def switches(rows):
    scored = [r for r in rows if r.get('decision') is not None]
    return [b['round'] for a, b in zip(scored, scored[1:]) if a['strategy'] != b['strategy']]


def select_cases(records):
    """Deterministic behavior-only selection, with absent/duplicate patterns reported."""
    cases, selection = [], []
    grouped = defaultdict(list)
    for task, record in records.items():
        grouped[record['scenario']['id']].append(task)
    for scenario, tasks in sorted(grouped.items()):
        scores = {}
        for task in sorted(tasks):
            rec = records[task]
            seqs = [sequence(rec, a) for a in agents(rec)]
            by_round = defaultdict(list)
            for row in rec['rows']:
                by_round[row['round']].append(row['strategy'])
            longest = current = 0
            for _, choices in sorted(by_round.items()):
                current = current + 1 if len(set(choices)) > 1 else 0
                longest = max(longest, current)
            sustained = 0
            for seq in seqs:
                for index, row in enumerate(seq):
                    if row['round'] in switches(seq) and len(seq[index:]) >= 6 and len({x['strategy'] for x in seq[index:]}) == 1:
                        sustained = max(sustained, len(seq[index:]))
            scores[task] = (longest, sustained, sum(len(switches(s)) for s in seqs))
        candidates = []
        for index, reason in enumerate(('persistent_disagreement', 'sustained_switch', 'repeated_switching')):
            task = max(sorted(tasks), key=lambda t: scores[t][index])
            value = scores[task][index]
            candidates.append((reason, [task], value if value >= (2 if index == 2 else 1) else 0))
        pairs = []
        for a in sorted(tasks):
            for b in sorted(tasks):
                ca, cb = records[a]['config'], records[b]['config']
                if ca['schedule'] != 'synchronous' or cb['schedule'] != 'sequential':
                    continue
                if {k:v for k,v in ca.items() if k != 'schedule'} != {k:v for k,v in cb.items() if k != 'schedule'}:
                    continue
                aa = {(r['agent'], r['round']):r['strategy'] for r in records[a]['rows']}
                bb = {(r['agent'], r['round']):r['strategy'] for r in records[b]['rows']}
                if aa.keys() == bb.keys():
                    pairs.append((sum(aa[k] != bb[k] for k in aa), a, b))
        best = max(pairs, key=lambda x:x[0]) if pairs else (0, '', '')
        candidates.append(('schedule_sensitivity', list(best[1:]), best[0]))
        seen = {}
        for reason, chosen, score in candidates:
            key = tuple(chosen)
            if not score:
                status, case_id = 'absent', ''
            elif key in seen:
                status, case_id = 'duplicate', seen[key]
            else:
                case_id = f'analysis_case{len(cases)+1:02d}'
                seen[key] = case_id
                cases.append(dict(id=case_id, scenario=scenario, reason=reason, tasks=chosen))
                status = 'selected'
            selection.append(dict(scenario=scenario, reason=reason, score=score, status=status, case=case_id, tasks=';'.join(chosen) if score else ''))
    return cases, selection


def within_plot(root, case, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [(t, a) for t in case['tasks'] for a in agents(records[t])]
    fig, axes = plt.subplots(len(case['tasks']), 2, figsize=(13, 4*len(case['tasks'])), squeeze=False, layout='constrained')
    data = []
    for ax, (task, agent) in zip(axes.flat, panels):
        seq = sequence(records[task], agent)
        changes = switches(seq)
        for row in seq:
            vals = [(row.get(k) or {}).get('margin') for k in ('before', 'decision', 'post')]
            xs = [row['round'] + offset for offset in (-.23, 0, .23)]
            ax.plot(xs, [float('nan') if v is None else v for v in vals], color='0.65', linewidth=1)
            for x, value, color, marker in zip(xs, vals, ('#777777', '#1764ab', '#bb3388'), ('o', 's', '^')):
                if value is not None:
                    ax.scatter(x, value, color=color, marker=marker, s=15)
            if row['round'] in changes:
                ax.axvline(row['round'], color='orange', alpha=.5)
            ax.text(row['round'], .02, str(row['strategy']), transform=ax.get_xaxis_transform(), ha='center', fontsize=7)
            data.append(dict(task=task, agent=agent, round=row['round'], strategy=row['strategy'], forced=row.get('forced',False), before=vals[0], decision=vals[1], post=vals[2], scored_switch=row['round'] in changes, sentiment=row.get('sentiment')))
        ax.axhline(0, color='black', linewidth=.6)
        ax.set(title=f"{agent} · {records[task]['config']['schedule']}\n{seq[0]['model']}", xlabel='Round (bottom labels: public strategy)', ylabel='Raw likelihood margin')
    fig.suptitle(case['reason']+' — gray: before; blue: decision; magenta: post; orange: switch')
    fig.savefig(root/'figures'/f"{case['id']}_within_step.png", dpi=160)
    plt.close(fig)
    csv_write(root/'tables'/f"{case['id']}_within_step.csv", data)


def replay_row(runtime, row, scenario, tolerance=.01):
    """Independent residual-output patches from before into after peer delivery."""
    from .interventions import replacement
    after = row['decision_history']
    count = len(row['deliveries'])
    if not count or row['round'] == 0:
        return []
    before = after[:-count]
    if any(m['role'] != 'user' for m in after[-count:]):
        raise ValueError('Cannot identify trailing peer deliveries')
    probe = row['decision']['probe']
    texts = {'stance_probe':probe.replace('{','{{').replace('}','}}')}
    baseline = runtime.measure(after, scenario, texts)
    if baseline['rendered_prompt'] != row['decision']['rendered_prompt']:
        raise ValueError('Replay prompt differs from saved decision prompt')
    error = baseline['margin'] - row['decision']['margin']
    if not math.isfinite(error) or abs(error) > tolerance:
        raise ValueError(f'Replay baseline differs by {error:g}, exceeding {tolerance:g}')
    if (baseline['margin'] >= 0) != (row['decision']['margin'] >= 0):
        raise ValueError('Replay changes the saved decision sign')
    blocks = runtime.model.model.layers
    vectors, positions = [], []
    for history in (before, after):
        context = history + [{'role':'user','content':probe}]
        position = len(runtime.tokenizer(runtime.inputs(context)[0])['input_ids']) - 1
        positions.append(position)
        capture, handles = {}, []
        def hook_for(layer):
            def hook(module, inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                capture[layer] = tensor[0, position].detach().clone()
            return hook
        try:
            for layer, block in enumerate(blocks):
                handles.append(block.register_forward_hook(hook_for(layer)))
            runtime.likelihood(context, scenario['strategy_1']['candidate'])
        finally:
            for handle in handles:
                handle.remove()
        vectors.append(capture)
    results = []
    for layer, block in enumerate(blocks):
        donor, recipient = vectors[0][layer], vectors[1][layer]
        with replacement(block, positions[1], recipient):
            self_margin = runtime.measure(after, scenario, texts)['margin']
        if not math.isfinite(self_margin) or abs(self_margin-baseline['margin']) > tolerance:
            raise ValueError(f'Self-patch check failed at layer {layer}')
        with replacement(block, positions[1], donor):
            patched = runtime.measure(after, scenario, texts)['margin']
        if not math.isfinite(patched):
            raise ValueError(f'Non-finite patched margin at layer {layer}')
        d, r = donor.float(), recipient.float()
        results.append(dict(agent=row['agent'], round=row['round'], layer=layer,
            activation_change=(r-d).norm().item()/max(d.norm().item(),1e-12),
            baseline=baseline['margin'], saved_baseline=row['decision']['margin'], replay_error=error,
            patched_margin=patched, delta=patched-baseline['margin'],
            choice_flip=(patched>=0)!=(baseline['margin']>=0), self_patch_error=self_margin-baseline['margin'],
            donor_position=positions[0], recipient_position=positions[1], anchor='final_prompt_token', component='block_residual_output'))
        print(f"  {row['agent']} round {row['round']} layer {layer+1}/{len(blocks)}", flush=True)
    return results


def heatmaps(root, case, records, data):
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import matplotlib.pyplot as plt
    panels = [(t,a) for t in case['tasks'] for a in agents(records[t])]
    for field, suffix, label in [('activation_change','activation_change','Relative residual change ||after-before|| / ||before||'), ('delta','intervention_effect','Patched minus unmodified decision margin')]:
        fig, axes = plt.subplots(len(case['tasks']), 2, figsize=(13,5*len(case['tasks'])), squeeze=False, layout='constrained')
        limit = max([abs(d[field]) for d in data] or [1]) or 1
        for ax, (task, agent) in zip(axes.flat, panels):
            subset = [d for d in data if d['task']==task and d['agent']==agent]
            rounds = sorted({d['round'] for d in subset})
            layers = sorted({d['layer'] for d in subset})
            if not subset:
                ax.set_visible(False)
                continue
            matrix = np.full((len(layers),len(rounds)), np.nan)
            for d in subset:
                y,x = layers.index(d['layer']),rounds.index(d['round'])
                matrix[y,x] = d[field]
                if field=='delta' and d['choice_flip']:
                    ax.plot(x,y,'kx',markersize=4)
            img=ax.imshow(matrix, origin='lower', aspect='auto', cmap='coolwarm' if field=='delta' else 'viridis', vmin=-limit if field=='delta' else 0, vmax=limit)
            ax.set_xticks(range(len(rounds)),rounds,fontsize=7)
            ax.set(title=f"{agent} · {records[task]['config']['schedule']}\n{sequence(records[task], agent)[0]['model']}",xlabel='Round',ylabel='Transformer block (zero-based)')
            fig.colorbar(img,ax=ax,label=label)
        fig.suptitle(case['reason'] + (' — ×: public choice would flip' if field=='delta' else ''))
        fig.savefig(root/'figures'/f"{case['id']}_{suffix}.png",dpi=160)
        plt.close(fig)


def execute(root, stage='behavioral', device='auto', tolerance=.01):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be nonnegative')
    root = Path(root)
    files = sorted((root/'conversations').glob('*.json'))
    if not files:
        raise ValueError('No saved conversations found')
    records = {f.stem:json.loads(f.read_text()) for f in files}
    for record in records.values():
        if len(agents(record)) != 2:
            raise ValueError('Experiment C analysis requires two agents')
        for agent in agents(record):
            seq = sequence(record, agent)
            if [r['round'] for r in seq] != list(range(record['config']['rounds']+1)):
                raise ValueError('Incomplete or duplicate conversation rounds')
    for folder in ('figures','tables'):
        (root/folder).mkdir(exist_ok=True)
    cases, selection = select_cases(records)
    manifest = dict(analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), version=1, stage=stage, cases=cases, source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        donor='same agent, same checkpoint, before latest peer delivery', component='residual output of each block independently', anchor='final prompt token', tolerance=tolerance,
        note='Observational activation change; causal patch effect at one token site, not a decomposition or sentiment intervention. No dialogue continuation. Forced round zero has no decision score.')
    write(root/'tables/analysis_manifest.json',manifest)
    csv_write(root/'tables/analysis_selection.csv',selection)
    cache = {}
    runtime = None
    try:
        for case in cases:
            print(f"{case['id']}: {case['reason']} ({', '.join(case['tasks'])})",flush=True)
            within_plot(root,case,records)
            if stage != 'all':
                continue
            data = []
            for task in case['tasks']:
                rec = records[task]
                for row in sorted(rec['rows'], key=lambda r:(r['model'], r['agent'], r['round'])):
                    if row['round']==0 or not row.get('deliveries'):
                        continue
                    key=(task,row['agent'],row['round'])
                    if key not in cache:
                        if runtime is None or runtime.name != row['model']:
                            # Grouping is deliberately unnecessary for correctness; release old weights.
                            if runtime is not None:
                                del runtime
                            from .runtime import Runtime
                            runtime=Runtime(row['model'],device)
                        cache[key]=[dict(task=task,**d) for d in replay_row(runtime,row,rec['scenario'],tolerance)]
                    data.extend(cache[key])
            csv_write(root/'tables'/f"{case['id']}_layers.csv",data)
            heatmaps(root,case,records,data)
        write(root/'tables/analysis_status.json',dict(phase='complete',stage=stage,cases=len(cases)))
    except BaseException as error:
        write(root/'tables/analysis_status.json',dict(phase='failed',error=str(error)))
        raise
    print(f"Analysis complete: {root/'figures'}",flush=True)
    return cases
