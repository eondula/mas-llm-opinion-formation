#!/usr/bin/env python3
"""Evaluate saved held-out stance forecasts as answer and consensus forecasts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from model_saved_pilot import export, hashes


def answer(stance):
    return 'Yes' if stance > 0 else 'No' if stance < 0 else 'Tie'


def group_answer(answers):
    if any(value not in ('Yes', 'No') for value in answers):
        return 'Unknown'
    return answers[0] if len(set(answers)) == 1 else 'Split'


def observations(source):
    records = {}
    for path in sorted((source/'conversations').glob('*.json')):
        record = json.loads(path.read_text())
        label = record.get('evaluation', {}).get('reference_answer')
        for row in record['rows']:
            records[(path.stem, row['response_id'])] = dict(
                question=record['question_id'], conversation=path.stem, response_id=row['response_id'],
                agent=row['agent'], model=row['model'], round=row['round'], schedule=record['config']['schedule'],
                composition=' | '.join(record['config']['models']), label=label,
                explicit=row['answer'] if row.get('choice_status')=='valid' and row.get('answer') in ('Yes','No') else 'Invalid',
                stance_answer=answer(row['post']['stance']))
    return records


def build(predictions, observed):
    agents = []
    for prediction in predictions:
        actual = observed[(prediction['conversation'], prediction['response_id'])]
        agents.append(dict(actual, subset=prediction['subset'], method=prediction['method'], mode=prediction['mode'],
            predicted=answer(prediction['prediction'])))
    initial = {(r['conversation'], r['agent']): r for r in observed.values() if r['round']==0}
    for actual in observed.values():
        if actual['round']==0:
            continue
        for method, predicted in (('initial_explicit', initial[(actual['conversation'],actual['agent'])]['explicit']),
                                  ('always_yes', 'Yes'), ('always_no', 'No')):
            agents.append(dict(actual, subset='all_finite', method=method, mode='fixed_baseline', predicted=predicted))
    expected = defaultdict(int)
    for actual in observed.values():
        expected[(actual['conversation'], actual['round'])] += 1
    buckets = defaultdict(list)
    for row in agents:
        buckets[(row['conversation'],row['round'],row['subset'],row['method'],row['mode'])].append(row)
    for conversation, turn, subset, method, mode in list(buckets):
        if subset=='all_finite' and method not in ('initial_explicit','always_yes','always_no'):
            buckets.setdefault((conversation,turn,'valid_only',method,mode), [])
    observed_groups = {}
    for row in observed.values():
        observed_groups[(row['conversation'],row['round'])] = row
    groups = []
    for (conversation, turn, subset, method, mode), rows in sorted(buckets.items()):
        # Sensitivity filtering must never turn an incomplete group into consensus.
        complete = len(rows) == expected[(conversation,turn)]
        first = rows[0] if rows else observed_groups[(conversation,turn)]
        groups.append(dict(question=first['question'],conversation=conversation,round=turn,
            schedule=first['schedule'],composition=first['composition'],label=first['label'],subset=subset,method=method,mode=mode,
            complete=complete,agents=len(rows),expected_agents=expected[(conversation,turn)],
            predicted=group_answer([r['predicted'] for r in rows]) if complete else 'Unknown',
            explicit=group_answer([r['explicit'] for r in rows]) if complete else 'Unknown',
            stance_answer=group_answer([r['stance_answer'] for r in rows]) if complete else 'Unknown'))
    return agents, groups


def macro(rows, function):
    by_question = defaultdict(list)
    for row in rows:
        value = function(row)
        if value is not None:
            by_question[row['question']].append(float(value))
    return sum(sum(values)/len(values) for values in by_question.values())/len(by_question) if by_question else None


def summarize(rows, level):
    keys = ('schedule','subset','method','mode','round') + (('model',) if level=='agent' else ())
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in keys)].append(row)
    output = []
    for values, selected in sorted(buckets.items()):
        available = selected
        if level=='group':
            selected = [r for r in selected if r['complete']]
        valid = lambda r: r['explicit'] in ('Yes','No') if level=='agent' else r['complete'] and r['explicit']!='Unknown'
        labeled = lambda r: r['label'] in ('Yes','No')
        metrics = dict(zip(keys,values),rows=len(available),questions=len({r['question'] for r in selected}),
            explicit_evaluable_rows=sum(valid(r) for r in selected),
            explicit_match_valid=macro(selected,lambda r: r['predicted']==r['explicit'] if valid(r) else None),
            explicit_match_strict=macro(selected,lambda r: r['predicted']==r['explicit'] if valid(r) else False),
            stance_sign_match=macro(selected,lambda r: r['predicted']==r['stance_answer'] if level=='agent' or r['complete'] else None),
            predicted_correct=macro(selected,lambda r: r['predicted']==r['label'] if labeled(r) else None),
            observed_explicit_correct=macro(selected,lambda r: r['explicit']==r['label'] if labeled(r) else None))
        if level=='group':
            metrics.update(complete_groups=sum(r['complete'] for r in selected),
                predicted_consensus_rate=macro(selected,lambda r:r['predicted'] in ('Yes','No') if r['complete'] else None),
                observed_consensus_rate=macro(selected,lambda r:r['explicit'] in ('Yes','No') if valid(r) else None),
                consensus_detection_accuracy=macro(selected,lambda r:(r['predicted'] in ('Yes','No'))==(r['explicit'] in ('Yes','No')) if valid(r) else None),
                consensus_precision=macro(selected,lambda r:r['explicit'] in ('Yes','No') if valid(r) and r['predicted'] in ('Yes','No') else None),
                consensus_recall=macro(selected,lambda r:r['predicted'] in ('Yes','No') if valid(r) and r['explicit'] in ('Yes','No') else None))
        output.append(metrics)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/experiments/new_exp_1_a'))
    parser.add_argument('--modeling',type=Path,default=Path('results/modeling/new_exp_1_a'))
    parser.add_argument('--output',type=Path,default=Path('results/modeling/new_exp_1_a_outcomes'))
    args=parser.parse_args()
    source,modeling,output=(p.resolve() for p in (args.source,args.modeling,args.output))
    for protected in (source,modeling):
        if output==protected or protected in output.parents or output in protected.parents:
            parser.error('Output must be separate from both input directories')
    if output.exists():
        parser.error('Choose a fresh output directory')
    source_hashes,model_hashes=hashes(source),hashes(modeling)
    manifest=json.loads((modeling/'manifest.json').read_text())
    if source_hashes!=manifest['source_sha256']:
        parser.error('Experiment does not match the modeling source manifest')
    predictions=json.loads((modeling/'predictions.json').read_text())
    observed=observations(source)
    agents,groups=build(predictions,observed)
    agent_summary,group_summary=summarize(agents,'agent'),summarize(groups,'group')
    last={}
    for r in observed.values():
        last[r['conversation']]=max(last.get(r['conversation'],0),r['round'])
    final_agents=[r for r in agents if r['round']==last[r['conversation']]]
    final_groups=[r for r in groups if r['round']==last[r['conversation']]]
    # Confusion counts preserve question identity so aggregation remains auditable.
    confusion=defaultdict(int)
    for r in groups:
        confusion[(r['question'],r['schedule'],r['subset'],r['method'],r['mode'],r['round'],r['explicit'],r['predicted'])]+=1
    confusion_rows=[dict(zip(('question','schedule','subset','method','mode','round','observed','predicted'),key),count=count) for key,count in sorted(confusion.items())]
    if hashes(source)!=source_hashes or hashes(modeling)!=model_hashes:
        raise RuntimeError('Input changed during evaluation')
    output.mkdir(parents=True)
    for name,rows in (('agent_predictions',agents),('group_predictions',groups),('agent_by_round',agent_summary),('group_by_round',group_summary),
                      ('agent_final',summarize(final_agents,'agent')),('group_final',summarize(final_groups,'group')),('group_confusion',confusion_rows)):
        export(output,name,rows)
    (output/'manifest.json').write_text(json.dumps(dict(source=str(source),modeling=str(modeling),source_sha256=source_hashes,
        modeling_sha256=model_hashes,inputs_unchanged=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).with_name('model_saved_pilot.py').read_bytes()).hexdigest(),
        definitions=dict(sign='positive Yes; negative No; exact zero Tie; no fitted threshold',
            group='unanimous Yes / unanimous No / Split; invalid or incomplete groups Unknown',
            correctness='evaluator labels only; split/unknown/invalid predictions count unsuccessful; missing labels excluded',
            validation='inherits saved leave-one-question-out predictions; labels never used for prediction',
            averaging='question macro averages; consensus precision/recall average only questions with relevant denominators',
            sensitivity='valid_only retains modeling eligibility; group metrics use complete groups only; consult coverage',
            strict_match='invalid explicit outputs count as unsuccessful; valid match excludes them',
            final='last recorded update per conversation; no extrapolation'),
        limitations=['Only ten question clusters; no causal interpretation.','Stance sign can differ from explicit answer.',
            'Correctness of predicted answers is distinct from forecasting actual agent behavior.',
            'Group correctness means correct unanimous consensus, not majority accuracy for larger groups.']),indent=2)+'\n')
    print(output)


if __name__=='__main__':
    main()
