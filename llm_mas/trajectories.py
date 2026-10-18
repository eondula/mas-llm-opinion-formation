"""Read old and current conversation records without rewriting scientific data."""
from pathlib import Path
from .storage import read

SCHEMA_VERSION = 2


def read_conversation(path):
    record = read(path)
    if record.get('schema_version', 1) not in (1, SCHEMA_VERSION):
        raise ValueError('Unsupported conversation schema version')
    if not isinstance(record.get('rows'), list):
        raise ValueError('Conversation must contain response rows')
    return record


def conversations(path):
    for file in sorted((Path(path) / 'conversations').glob('*.json')):
        yield file, read_conversation(file)


def observations(record, task=None):
    """Consistent identities/stages; scores and generated answers stay separate."""
    scenario = record.get('scenario', {})
    config = record.get('config', {})
    for row in record['rows']:
        yield dict(task=task, question_id=record.get('question_id', scenario.get('id')),
                   repetition=record.get('repetition', config.get('repetition', 0)),
                   response_id=row.get('response_id'), agent=row['agent'],
                   model=row.get('model'), round=row['round'],
                   answer=row.get('answer', row.get('strategy')), answer_status=row.get('choice_status'),
                   before=row.get('before'), decision=row.get('decision'),
                   decision_time=row.get('decision_time'), stance_source=row.get('stance_source','post_response_probe'), post=row.get('post'), sentiment=row.get('sentiment'),
                   argument=row.get('argument'), deliveries=row.get('deliveries', []))
