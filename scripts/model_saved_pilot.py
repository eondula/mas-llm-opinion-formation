#!/usr/bin/env python3
"""Offline stance dynamics; no inference imports and no writes to source experiments."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

MODELS = ('persistence', 'control_drift', 'degroot', 'friedkin_johnsen')


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def export(root, name, rows):
    (root / (name + '.json')).write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    if rows:
        with (root / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def extract(source):
    examples = []
    for path in sorted((source / 'conversations').glob('*.json')):
        record = json.loads(path.read_text())
        rows = record['rows']
        by_id = {r['response_id']: r for r in rows}
        previous = {}
        initial = {r['agent']: r for r in rows if r['round'] == 0}
        for index, row in enumerate(rows):
            if row['round'] == 0:
                previous[row['agent']] = row
                continue
            prior = previous[row['agent']]
            peers = []
            for delivery in row.get('deliveries', []):
                peer = by_id[delivery['source_id']]
                if rows.index(peer) >= index or peer['message'] != delivery['message']:
                    raise ValueError(f'Invalid delivery timing/content: {path}:{row["response_id"]}')
                peers.append(peer)
            if record['config']['schedule'] == 'no_peer' and peers:
                raise ValueError('Control contains peer deliveries')
            first = initial[row['agent']]
            states = [prior, first, row] + peers
            valid = all(r.get('choice_status') == 'valid' for r in states)
            values = [r['post']['stance'] for r in states]
            if not all(np.isfinite(v) for v in values):
                raise ValueError('Nonfinite stance; explicit handling required')
            examples.append(dict(conversation=path.stem, question=record['question_id'],
                pair_id=record.get('pair_id'), repetition=record.get('repetition'),
                composition=' | '.join(record['config']['models']),
                initialization=str(record['config']['initialization']), schedule=record['config']['schedule'],
                model=row['model'], agent=row['agent'], round=row['round'], response_id=row['response_id'],
                prior_id=prior['response_id'], peer_ids='|'.join(p['response_id'] for p in peers),
                own=float(prior['post']['stance']), initial=float(first['post']['stance']),
                peer=float(np.mean([p['post']['stance'] for p in peers])) if peers else float(prior['post']['stance']),
                target=float(row['post']['stance']), valid=valid,
                current_valid=row.get('choice_status') == 'valid', peer_count=len(peers)))
            previous[row['agent']] = row
    if not examples:
        raise ValueError('No saved updates found')
    return examples


def simplex_fit(x, y):
    """Exact least squares over the three-weight simplex, including all faces."""
    best = None
    for size in range(1, x.shape[1] + 1):
        for active in itertools.combinations(range(x.shape[1]), size):
            a = x[:, active]
            kkt = np.block([[a.T @ a, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
            solution = np.linalg.lstsq(kkt, np.r_[a.T @ y, 1.], rcond=None)[0][:size]
            if np.min(solution) < -1e-8:
                continue
            weights = np.zeros(x.shape[1]); weights[list(active)] = np.maximum(solution, 0)
            weights /= weights.sum()
            loss = np.mean((x @ weights - y) ** 2)
            if best is None or loss < best[0]:
                best = (loss, weights)
    return best[1]


def fit(kind, rows):
    y = np.array([r['target'] for r in rows])
    own = np.array([r['own'] for r in rows])
    if kind == 'persistence':
        return dict(intercept=0., own=1., peer=0., initial=0.)
    if kind == 'control_drift':
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(rows)), own]), y, rcond=None)[0]
        return dict(intercept=float(coef[0]), own=float(coef[1]), peer=0., initial=0.)
    peer = np.array([r['peer'] for r in rows])
    if kind == 'degroot':
        gap = peer - own
        alpha = float(np.clip(gap @ (y - own) / (gap @ gap), 0, 1)) if gap @ gap > 1e-12 else 0.
        return dict(intercept=0., own=1-alpha, peer=alpha, initial=0.)
    weights = simplex_fit(np.column_stack([own, peer, [r['initial'] for r in rows]]), y)
    return dict(intercept=0., own=float(weights[0]), peer=float(weights[1]), initial=float(weights[2]))


def predict(parameters, own, peer, initial):
    return float(np.clip(parameters['intercept'] + parameters['own']*own + parameters['peer']*peer + parameters['initial']*initial, -1, 1))


def evaluate(examples):
    predictions, coefficients = [], []
    questions = sorted({r['question'] for r in examples})
    if len(questions) < 3:
        raise ValueError('At least three distinct questions required')
    for heldout in questions:
        for model, schedule in sorted({(r['model'], r['schedule']) for r in examples}):
            for subset in ('all_finite', 'valid_only'):
                training = [r for r in examples if r['question'] != heldout and r['model'] == model and (subset == 'all_finite' or r['valid'])]
                test = [r for r in examples if r['question'] == heldout and r['model'] == model and r['schedule'] == schedule]
                for kind in MODELS:
                    train = [r for r in training if r['schedule'] == ('no_peer' if kind == 'control_drift' else schedule)]
                    if not train:
                        raise ValueError(f'Missing training data: {model}, {schedule}, {subset}')
                    parameters = fit(kind, train)
                    coefficients.append(dict(heldout_question=heldout, model=model, schedule=schedule, subset=subset,
                        method=kind, train_rows=len(train), train_questions=len({r['question'] for r in train}), **parameters))
                    # Roll out all agents jointly, including the other checkpoint in mixed groups.
                    # Parameters for each other agent are fitted separately below during joint rollout.
                    for row in test:
                        if subset == 'valid_only' and not row['valid']:
                            continue
                        value = predict(parameters, row['own'], row['peer'], row['initial'])
                        predictions.append(dict(question=heldout, conversation=row['conversation'], response_id=row['response_id'],
                            model=model, schedule=schedule, subset=subset, method=kind, round=row['round'],
                            target=row['target'], prediction=value, valid=row['valid'], mode='one_step'))
    lookup = {(c['heldout_question'], c['model'], c['schedule'], c['subset'], c['method']): c for c in coefficients}
    for subset in ('all_finite', 'valid_only'):
        for kind in MODELS:
            state = {(r['conversation'], r['prior_id']): r['own'] for r in examples if r['round'] == 1}
            for row in examples:
                key = (row['conversation'], row['response_id'])
                own = state.get((row['conversation'], row['prior_id']), row['initial'])
                peer = np.mean([state[(row['conversation'], pid)] for pid in row['peer_ids'].split('|')]) if row['peer_ids'] else own
                parameters = lookup[(row['question'], row['model'], row['schedule'], subset, kind)]
                value = predict(parameters, own, peer, row['initial'])
                state[key] = value
                # Initial source rows are fixed observed starts, inserted before updates below.
                if subset == 'all_finite' or row['valid']:
                    predictions.append(dict(question=row['question'], conversation=row['conversation'], response_id=row['response_id'],
                        model=row['model'], schedule=row['schedule'], subset=subset, method=kind, round=row['round'],
                        target=row['target'], prediction=value, valid=row['valid'], mode='recursive'))
    return predictions, coefficients


def summarize(predictions):
    output = []
    keys = ('model', 'schedule', 'subset', 'method', 'mode')
    for values in sorted({tuple(r[k] for k in keys) for r in predictions}):
        selected = [r for r in predictions if tuple(r[k] for k in keys) == values]
        errors = []
        for question in sorted({r['question'] for r in selected}):
            rows = [r for r in selected if r['question'] == question]
            error = np.array([r['prediction'] - r['target'] for r in rows])
            errors.append((float(np.mean(abs(error))), float(np.mean(error**2))))
        output.append(dict(zip(keys, values), rows=len(selected), questions=len(errors),
            question_macro_mae=float(np.mean([e[0] for e in errors])),
            question_macro_rmse=float(np.sqrt(np.mean([e[1] for e in errors])))))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/experiments/new_exp_1_a'))
    parser.add_argument('--output', type=Path, default=Path('results/modeling/new_exp_1_a'))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error('Output must be separate from source experiment')
    if output.exists():
        parser.error('Output already exists; choose a new directory to preserve derived analyses')
    before = hashes(source)
    examples = extract(source)
    # Seed recursive state with the immutable round-zero post states.
    predictions, coefficients = evaluate(examples)
    summary = summarize(predictions)
    after = hashes(source)
    if before != after:
        raise RuntimeError('Source changed during analysis; outputs not written')
    output.mkdir(parents=True)
    export(output, 'modeling_rows', examples)
    export(output, 'predictions', predictions)
    export(output, 'fold_parameters', coefficients)
    export(output, 'summary', summary)
    stability = []
    for model, schedule, subset, method in sorted({(r['model'], r['schedule'], r['subset'], r['method']) for r in coefficients}):
        selected = [r for r in coefficients if (r['model'], r['schedule'], r['subset'], r['method']) == (model, schedule, subset, method)]
        for parameter in ('intercept', 'own', 'peer', 'initial'):
            values = np.array([r[parameter] for r in selected])
            stability.append(dict(model=model, schedule=schedule, subset=subset, method=method, parameter=parameter, mean=float(values.mean()), standard_deviation=float(values.std()), minimum=float(values.min()), maximum=float(values.max())))
    export(output, 'parameter_stability', stability)
    export(output, 'question_metrics', question_metrics(predictions))
    manifest = dict(source=str(source), source_sha256=before, source_unchanged=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), numpy=np.__version__, python=sys.version,
        target='post-turn stance; original tanh margin scale', split='leave-one-question-out; all branches/compositions/repetitions grouped',
        fitting='model-specific; condition-specific except control_drift trained only on no_peer',
        sensitivity='valid_only fits and scores rows whose current, prior, initial and delivered peer answers are valid; recursive paths still advance every turn',
        limitations=['Exploratory ten-question pilot; weights are descriptive, not causal.',
            'Cross-checkpoint stance scales are uncalibrated.', 'Recursive forecasts generate stances only, not future messages.',
            'No hyperparameter selection; convex averaging weights constrained nonnegative and sum to one.',
            'No-peer DeGroot reduces to persistence; FJ own/peer weights are not separately identifiable in no-peer conditions.'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    plot(output, summary)
    print(output)


def question_metrics(predictions):
    output = []
    keys = ('question', 'model', 'schedule', 'subset', 'method', 'mode')
    buckets = {}
    for row in predictions:
        buckets.setdefault(tuple(row[k] for k in keys), []).append(row['prediction']-row['target'])
    for values, errors in sorted(buckets.items()):
        error = np.array(errors)
        output.append(dict(zip(keys, values), rows=len(error), mae=float(np.mean(abs(error))), mse=float(np.mean(error**2))))
    return output


def plot(output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [r for r in summary if r['subset']=='all_finite' and r['schedule']=='synchronous']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for axis, mode in zip(axes, ('one_step', 'recursive')):
        for model in sorted({r['model'] for r in selected}):
            axis.plot(MODELS, [next(r['question_macro_mae'] for r in selected if r['model']==model and r['method']==kind and r['mode']==mode) for kind in MODELS], marker='o', label=model.split('/')[1])
        axis.set_title(mode.replace('_', ' ')); axis.set_ylabel('Held-out question macro MAE'); axis.tick_params(axis='x', rotation=25); axis.legend()
    fig.savefig(output/'prediction_error.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
