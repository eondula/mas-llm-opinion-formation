"""Post-simulation task evaluation. Labels never enter the conversation engine."""
import csv
from .storage import write
from .trajectories import conversations


def evaluate(path, labels):
    rows, groups = [], []
    for file, record in conversations(path):
        qid = record['question_id']
        expected = labels.get(qid)
        previous = {}
        for row in record['rows']:
            answer = row.get('answer')
            correct = answer == expected if expected is not None and answer is not None else None
            old = previous.get(row['agent'])
            row['correct'] = correct
            rows.append(dict(task=file.stem, question_id=qid, repetition=record['repetition'],
                             research_split=record.get('research_split'), source_partition=record['dataset']['partition'],
                             agent=row['agent'], model=row['model'], round=row['round'],
                             answer=answer, reference_answer=expected, correct=correct,
                             valid=answer is not None, labeled=expected is not None,
                             stance=row['post']['stance'], margin=row['post']['margin'],
                             answer_reversal=old is not None and old[0] is not None and answer is not None and old[0] != answer,
                             correctness_transition=(('correct' if old[1] else 'incorrect') + '_to_' + ('correct' if correct else 'incorrect'))
                             if old is not None and old[1] is not None and correct is not None else None))
            previous[row['agent']] = (answer, correct)
        for t in sorted({r['round'] for r in record['rows']}):
            current = [r for r in record['rows'] if r['round'] == t]
            yes = sum(r.get('answer') == 'Yes' for r in current)
            no = sum(r.get('answer') == 'No' for r in current)
            complete = yes + no == record['config']['agents']
            majority = ('Yes' if yes > no else 'No') if yes != no else None
            groups.append(dict(task=file.stem, question_id=qid, round=t, yes=yes, no=no,
                               invalid=len(current)-yes-no, consensus=complete and (yes == len(current) or no == len(current)),
                               majority=majority if complete else None,
                               majority_correct=majority == expected if complete and majority and expected is not None else None))
        record['evaluation'] = dict(labeled=expected is not None, reference_answer=expected)
        write(file, record)
    folder = path / 'tables'
    folder.mkdir(exist_ok=True)
    for filename, data in [('correctness', rows), ('question_outcomes', groups)]:
        with (folder / (filename + '.csv')).open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(data[0]) if data else ['question_id'])
            writer.writeheader(); writer.writerows(data)
    labeled = [r for r in rows if r['labeled']]
    valid = [r for r in labeled if r['valid']]
    summary = dict(responses=len(rows), invalid_responses=sum(not r['valid'] for r in rows),
                   labeled_responses=len(labeled), correct=sum(r['correct'] is True for r in labeled),
                   accuracy=sum(r['correct'] is True for r in labeled)/len(labeled) if labeled else None,
                   valid_answer_accuracy=sum(r['correct'] is True for r in valid)/len(valid) if valid else None,
                   denominator='All labeled responses, including invalid outputs',
                   ground_truth_use='Post-simulation evaluation only')
    write(path / 'analysis/evaluation.json', dict(summary=summary, rows=rows, groups=groups))
    return summary
