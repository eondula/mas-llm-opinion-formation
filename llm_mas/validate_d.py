"""Standalone saved-run stance checks and explanation similarity; no sentiment.

python -m llm_mas.validate_d --dry-run
"""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
from .config import ROOT

ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
PARAPHRASES = {
    'sensor_tradeoff_001': [
        ['I favor higher measurement precision with narrower spatial coverage.',
         'I favor broader spatial coverage with lower measurement precision.'],
        ['I prefer the option that covers less area but measures more precisely.',
         'I prefer the option that covers more area but measures less precisely.']],
    'storage_tradeoff_001': [
        ['I favor higher data fidelity with greater storage requirements.',
         'I favor lower storage requirements with lower data fidelity.'],
        ['I prefer the option that preserves higher fidelity and uses more storage.',
         'I prefer the option that uses less storage and preserves lower fidelity.']],
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def probe(scenario, templates, candidates=None, order=(1, 2)):
    """Move whole blocks, preserving semantic strategy labels and argument pools."""
    candidates = candidates or [scenario[f'strategy_{k}']['candidate'] for k in (1, 2)]
    q = templates['combined_reasoning'].format(
        candidate_1=candidates[0], candidate_2=candidates[1],
        arguments_1='\n'.join('- '+s for s in scenario['strategy_1']['arguments']),
        arguments_2='\n'.join('- '+s for s in scenario['strategy_2']['arguments']))
    if order == (2, 1):
        start = q.index('STRATEGY 1\n')
        middle = q.index('STRATEGY 2\n', start)
        end = q.index('Return the exact candidate statement', middle)
        q = q[:start] + q[middle:end].rstrip() + '\n\n' + q[start:middle].rstrip() + '\n\n' + q[end:]
    return q


def score(runtime, history, candidates, instruction):
    parts = [runtime.likelihood(history + [{'role':'user', 'content':instruction}], c) for c in candidates]
    likelihoods = [p[0] for p in parts]
    counts = [len(p[1]) for p in parts]
    margin = likelihoods[0] - likelihoods[1]
    return dict(margin=margin, stance=math.tanh(margin/2), likelihoods=likelihoods,
                token_counts=counts, normalized_margin=likelihoods[0]/counts[0]-likelihoods[1]/counts[1],
                probe=instruction, rendered_prompt=parts[0][2])


def choice(text):
    # A strict first-line format prevents silently repairing invalid outputs.
    line = text.strip().splitlines()[0] if text.strip() else ''
    return line if line in ('A', 'B') else None


def stance_event(runtime, record, row, templates, tolerance):
    scenario = record['scenario']
    history = row['history']  # Validate the plotted post-response readout, including forced round 0.
    candidates = [scenario[f'strategy_{k}']['candidate'] for k in (1, 2)]
    original = row['post']['probe']
    baseline = score(runtime, history, candidates, original)
    repeat = score(runtime, history, candidates, original)
    variants = {'reversed_blocks':score(runtime, history, candidates, probe(scenario, templates, order=(2, 1)))}
    for i, pair in enumerate(PARAPHRASES[scenario['id']]):
        variants[f'paraphrase_{i+1}'] = score(runtime, history, pair, probe(scenario, templates, pair))
    independent = []
    for order in ((1, 2), (2, 1)):
        instruction = ('Choose between the alternatives below. Put exactly A or B on the first line, '
                       'then give a short explanation.\n' + '\n'.join(
                           f'{label}: {candidates[k-1]}' for label, k in zip(('A', 'B'), order)))
        output = runtime.generate(history+[{'role':'user','content':instruction}], 128)
        label = choice(output['output'])
        independent.append(dict(order=order, instruction=instruction, **output,
                                semantic_strategy=order[0 if label == 'A' else 1] if label else None))
    return dict(history=history, baseline=baseline, repeat=repeat, variants=variants,
                independent=independent, saved_margin=row['post']['margin'],
                replay_error=abs(baseline['margin']-row['post']['margin']),
                replay_within_tolerance=abs(baseline['margin']-row['post']['margin']) <= tolerance,
                model_revision=getattr(runtime.model.config, '_commit_hash', None),
                tokenizer_revision=runtime.tokenizer.init_kwargs.get('_commit_hash'),
                note='Post-response elicitation is converging evidence, not independent ground truth.')


def similarities(records, embeddings):
    rows = []
    for task, record in records:
        prior = {}
        for row in sorted(record['rows'], key=lambda r:(r['round'], r['agent'])):
            text = row['argument'].strip()
            history = prior.setdefault(row['agent'], [])
            valid = [(t, e) for t, e in history if e in embeddings]
            best, matched, adjacent = None, None, None
            if text in embeddings and valid:
                def cosine(other):
                    return max(-1., min(1., sum(a*b for a,b in zip(embeddings[text], embeddings[other]))))
                matched, other = max(valid, key=lambda item:cosine(item[1]))
                best = cosine(other)
                if history[-1][1] in embeddings:
                    adjacent = cosine(history[-1][1])
            rows.append(dict(task=task, agent=row['agent'], round=row['round'],
                             similarity=best, matched_round=matched, adjacent_similarity=adjacent,
                             explanation=text, status='missing' if not text else ('no_prior' if not valid else 'scored')))
            history.append((row['round'], text))
    return rows


def repetition(records, out, device, revision):
    import torch
    from transformers import AutoTokenizer, AutoModel
    cache = out/'embeddings.json'
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=revision)
        model = AutoModel.from_pretrained(ENCODER, revision=revision).eval().to(device)
        unique = sorted({r['argument'].strip() for _,d in records for r in d['rows'] if r['argument'].strip()})
        embeddings, truncated = {}, []
        for i in range(0, len(unique), 32):
            batch = unique[i:i+32]
            truncated.extend(s for s in batch if len(tokenizer(s)['input_ids']) > 256)
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            with torch.inference_mode():
                tokens = model(**inputs).last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1)
                pooled = (tokens*mask).sum(1)/mask.sum(1).clamp(min=1)
                vectors = torch.nn.functional.normalize(pooled, dim=1).cpu().tolist()
            embeddings.update(zip(batch, vectors))
        data = dict(encoder=ENCODER, revision=getattr(model.config, '_commit_hash', None),
                    max_tokens=256, pooling='attention-masked mean, L2 normalized',
                    truncated=truncated, embeddings=embeddings)
        write(cache, data)
    rows = similarities(records, data['embeddings'])
    csv_write(out/'semantic_repetition.csv', rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'results/studies/experiment_D_v1')
    p.add_argument('--output', type=Path, default=ROOT/'results/validation/experiment_D_v1')
    p.add_argument('--mode', choices=['all','stance','repetition'], default='all')
    p.add_argument('--rounds', type=int, nargs='+', default=[0,1,10])
    p.add_argument('--device', choices=['cpu','cuda','mps'], default='cpu')
    p.add_argument('--encoder-revision', default='main')
    p.add_argument('--tolerance', type=float, default=.01)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out == source or source in out.parents:
        p.error('Output must be outside the source run')
    if args.tolerance < 0 or not math.isfinite(args.tolerance):
        p.error('Tolerance must be finite and nonnegative')
    files = sorted((source/'conversations').glob('*.json'))
    if not files:
        p.error('No saved conversations found')
    records = [(f.stem, json.loads(f.read_text())) for f in files]
    templates = json.loads((source/'inputs/prompts.json').read_text())
    events = [(task,d,r) for task,d in records for r in d['rows'] if r['round'] in args.rounds]
    if not events:
        p.error('No rows match selected rounds')
    for _,d,r in events:
        if d['config']['argument_mode'] != 'combined_reasoning' or d['scenario']['id'] not in PARAPHRASES:
            p.error('This script supports Experiment D combined_reasoning Sensor/Storage records')
        if probe(d['scenario'], templates) != r['post']['probe']:
            p.error('Saved probe does not match template reconstruction')
    manifest = dict(source=str(source), source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                    prompts_hash=digest(templates), script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    rounds=sorted(set(args.rounds)), stage='post', tolerance=args.tolerance, device=args.device,
                    paraphrases=PARAPHRASES, encoder=ENCODER, encoder_revision=args.encoder_revision,
                    contexts=[dict(task=t, agent=r['agent'], round=r['round']) for t,_,r in events])
    print(f'{len(records)} conversations; {len(events)} stance contexts; '
          f'{sum(len(d["rows"]) for _,d in records)} repetition rows', flush=True)
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()) != manifest:
        p.error('Output manifest differs; choose a fresh output directory')
    write(out/'manifest.json', manifest)
    if args.mode in ('all','stance'):
        from .runtime import Runtime
        summaries = []
        for model in sorted({r['model'] for _,_,r in events}):
            runtime = None
            for task,d,r in events:
                if r['model'] != model:
                    continue
                target = out/'stance'/f'{task}_{r["agent"].replace(" ", "_")}_{r["round"]}.json'
                if target.exists():
                    result = json.loads(target.read_text())
                else:
                    if runtime is None:
                        runtime = Runtime(model, args.device)
                    print(f'Stance: {task} {r["agent"]} round {r["round"]}', flush=True)
                    result = stance_event(runtime, d, r, templates, args.tolerance)
                    write(target, result)
                base = result['baseline']['margin']
                summaries.append(dict(task=task, agent=r['agent'], round=r['round'], model=model,
                    saved_margin=result['saved_margin'], replay_margin=base, replay_error=result['replay_error'],
                    replay_pass=result['replay_within_tolerance'],
                    repeat_error=abs(base-result['repeat']['margin']),
                    normalized_sign_agrees=(base>=0)==(result['baseline']['normalized_margin']>=0),
                    variant_sign_agreement=sum((v['margin']>=0)==(base>=0) for v in result['variants'].values())/len(result['variants']),
                    independent_valid=sum(v['semantic_strategy'] is not None for v in result['independent']),
                    independent_agree=sum(v['semantic_strategy']==(1 if base>=0 else 2) for v in result['independent'])))
            del runtime
            gc.collect()
        csv_write(out/'stance_summary.csv', summaries)
    if args.mode in ('all','repetition'):
        repetition(records, out, args.device, args.encoder_revision)
    print(f'Outputs: {out}', flush=True)


if __name__ == '__main__':
    main()
