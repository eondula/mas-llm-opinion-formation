"""Pinned StrategyQA loading, safe task construction and question-level splits."""
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path
from .tasks import question_task

SOURCE_URL = 'https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip'
RELEASE_SHA256 = '4911d85eb6721a93bed7645419df77e721808b32b9785dee14ad80e6249e0a90'
PARTITION_SHA256 = {
    'train': '728e28b7d6bdfe6da3a0a5f8c5923113a79063d199e960cfcc9f8a3ab34cd92c',
    'test': '268bec2605a45830ae7f1f3861d5ee2a4a10f4aeaca0358a076760b2eff3ed25',
}
SPLIT_VERSION = 'normalized-question-sha256-70-15-15-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def download(root):
    folder = Path(root) / 'data/strategyqa'
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read()
    if sha(payload) != RELEASE_SHA256:
        raise ValueError('StrategyQA release checksum changed; inspect before updating the pinned release')
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for partition in ('train', 'test'):
            name = f'strategyqa_{partition}.json'
            data = archive.read(name)
            # Validate before replacing an installed dataset file.
            validate_records(json.loads(data), partition)
            temporary = folder / (name + '.tmp')
            temporary.write_bytes(data)
            temporary.replace(folder / name)
    return describe(root)


def validate_records(records, partition):
    if not isinstance(records, list) or not records:
        raise ValueError('StrategyQA partition must be a nonempty JSON list')
    ids = set()
    for row in records:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                            for key in ('qid', 'question')):
            raise ValueError('StrategyQA rows require nonempty qid and question strings')
        if row['qid'] in ids:
            raise ValueError('Duplicate StrategyQA question ID: ' + row['qid'])
        ids.add(row['qid'])
        if 'answer' in row and type(row['answer']) is not bool:
            raise ValueError('StrategyQA answer labels must be Boolean')
        if partition == 'train' and 'answer' not in row:
            raise ValueError('The labeled StrategyQA train pool requires an answer for every question')


def load(root, spec=None):
    spec = dict(spec or {})
    if spec.get('name', 'strategyqa') != 'strategyqa':
        raise ValueError('Unsupported dataset')
    partition = spec.get('partition', 'train')
    if partition not in ('train', 'test'):
        raise ValueError('StrategyQA source partition must be train or test')
    folder = Path(spec.get('path', 'data/strategyqa'))
    if not folder.is_absolute():
        folder = Path(root) / folder
    file = folder / f'strategyqa_{partition}.json'
    if not file.is_file():
        raise ValueError('StrategyQA is not installed. Run: python -m llm_mas dataset download')
    data = file.read_bytes()
    if spec.get('source_sha256') and spec['source_sha256'] != sha(data):
        raise ValueError('StrategyQA file changed since the saved configuration')
    rows = json.loads(data)
    validate_records(rows, partition)
    metadata = dict(name='strategyqa', path=str(folder.relative_to(root)) if folder.is_relative_to(root) else str(folder),
                    partition=partition, source_url=SOURCE_URL, release_sha256=RELEASE_SHA256,
                    source_sha256=sha(data), release_verified=sha(data) == PARTITION_SHA256[partition],
                    split_version=SPLIT_VERSION)
    return rows, metadata


def split_assignments(rows, seed=42, partition='train'):
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('Split seed must be an integer from 0 to 4294967295')
    result = {}
    for row in rows:
        # Identical normalized questions, even with different IDs, never cross splits.
        normalized = ' '.join(row['question'].casefold().split())
        bucket = int(sha((str(seed) + '\n' + normalized).encode())[:16], 16) / 2**64
        result[row['qid']] = ('train' if bucket < .7 else 'validation' if bucket < .85 else 'test') if partition == 'train' else 'official_test'
    return result


def select(root, raw):
    spec = raw.get('dataset') or {}
    if isinstance(spec, str):
        spec = {'name': spec}
    rows, metadata = load(root, spec)
    split = spec.get('split', 'train' if metadata['partition'] == 'train' else 'official_test')
    if split not in ('train', 'validation', 'test', 'all', 'official_test'):
        raise ValueError('Unknown research split')
    if metadata['partition'] == 'test' and split not in ('all', 'official_test'):
        raise ValueError('The official test source is prediction-only; use split=official_test or all')
    if metadata['partition'] == 'train' and split == 'official_test':
        raise ValueError('official_test requires the official test source partition')
    split_seed = spec.get('split_seed', 42)
    assignments = split_assignments(rows, split_seed, metadata['partition'])
    pool = [row for row in rows if split == 'all' or assignments[row['qid']] == split]
    balance=raw.get('question_balance','random')
    if balance not in ('random','equal'):raise ValueError('Unknown question balance')
    if balance=='equal' and metadata['partition']!='train':raise ValueError('Balanced sampling requires labeled questions')
    requested = raw.get('question_ids')
    count = raw.get('question_count', 1)
    sample_seed = raw.get('question_seed', raw.get('seed', 42))
    if type(sample_seed) is not int or not 0 <= sample_seed <= 2**32 - 1:
        raise ValueError('Question sampling seed must be an unsigned 32-bit integer')
    if requested:
        if not isinstance(requested, list) or not all(isinstance(q, str) for q in requested) or len(set(requested)) != len(requested):
            raise ValueError('Question IDs must be a list of unique strings')
        lookup = {row['qid']: row for row in pool}
        if any(qid not in lookup for qid in requested):
            raise ValueError('A selected question is not in the chosen source partition/research split')
        selected = [lookup[qid] for qid in requested]
    else:
        if type(count) is not int or count < 1 or count > len(pool):
            raise ValueError(f'Question count must be between 1 and {len(pool)}')
        selected = sorted(pool, key=lambda row: row['qid'])
        random.Random(sample_seed).shuffle(selected)
        if balance=='equal':
            if count%2:raise ValueError('Equal label balance requires an even question count')
            yes=[r for r in selected if r['answer']];no=[r for r in selected if not r['answer']]
            if min(len(yes),len(no))<count//2:raise ValueError('Insufficient questions in a label stratum')
            selected=yes[:count//2]+no[:count//2];random.Random(sample_seed).shuffle(selected)
        else:selected = selected[:count]
    if balance=='equal' and sum(r['answer'] for r in selected)*2!=len(selected):
        raise ValueError('Explicit question IDs must have equal Yes/No labels')
    metadata.update(split=split, split_seed=split_seed)
    tasks = [question_task(row['qid'], row['question']) for row in selected]
    return tasks, metadata, assignments


def labels(root, spec):
    # Evaluator-only entry point. Neither engine nor prompt builders call this.
    rows, _ = load(root, spec)
    return {row['qid']: 'Yes' if row['answer'] else 'No' for row in rows if 'answer' in row}


def describe(root, partition='train', split='train', split_seed=42):
    rows, metadata = load(root, dict(partition=partition))
    assignments = split_assignments(rows, split_seed, partition)
    return dict(metadata=metadata, labeled=sum('answer' in r for r in rows), total=len(rows),
                splits={s: sum(value == s for value in assignments.values()) for s in sorted(set(assignments.values()))},
                questions=[dict(id=r['qid'], question=r['question']) for r in rows if split == 'all' or assignments[r['qid']] == split])
