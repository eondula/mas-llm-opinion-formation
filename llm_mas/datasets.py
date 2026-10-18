"""Pinned StrategyQA and BoolQ loading, safe task construction and question-level splits."""
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


def boolq_question(text):
    # BoolQ questions are lowercase search queries without punctuation.
    text = ' '.join(text.split())
    text = text[:1].upper()+text[1:]
    return text if text.endswith('?') else text+'?'


def boolq_partition(archive, partition):
    """SuperGLUE BoolQ JSONL -> StrategyQA-style records; passages are kept but never shown to agents."""
    rows = []
    for line in archive.read(f'BoolQ/{partition}.jsonl').decode().splitlines():
        item = json.loads(line)
        row = dict(qid=f'boolq-{partition}-{item["idx"]}', question=boolq_question(item['question']),
                   source_question=item['question'], passage=item['passage'])
        if 'label' in item:
            row['answer'] = item['label']
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False, separators=(',', ':')).encode()


# SuperGLUE v2 release of BoolQ (Clark et al., 2019); the original storage.googleapis.com/boolq files are no longer public.
DATASETS = {
    'strategyqa': dict(label='StrategyQA', source_url=SOURCE_URL, release_sha256=RELEASE_SHA256, partition_sha256=PARTITION_SHA256,
                       extract=lambda archive, partition: archive.read(f'strategyqa_{partition}.json')),
    'boolq': dict(label='BoolQ', source_url='https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip',
                  release_sha256='853fbe7922f70c59629f06a39e8d9ca440c3d740e760fd3b87a5ddf3dcba2436',
                  partition_sha256={'train': 'f0ba5ec63c4621c499b2d9725a7d2e7bf07977681ddb189d3f388abaa65a9a90', 'test': '08eee1227567396a0fedfd532a088037318406e0972ca2599c531ae38953f463'}, extract=boolq_partition),
}


def source(name):
    if name not in DATASETS:
        raise ValueError('Unsupported dataset')
    return DATASETS[name]


def download(root, name='strategyqa'):
    spec = source(name)
    folder = Path(root) / 'data' / name
    context = None
    try:
        # Python.org macOS builds ship without CA certificates; certifi is a pinned requirement.
        import certifi, ssl
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    with urllib.request.urlopen(spec['source_url'], timeout=60, context=context) as response:
        payload = response.read()
    if sha(payload) != spec['release_sha256']:
        raise ValueError(f'{spec["label"]} release checksum changed; inspect before updating the pinned release')
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for partition in ('train', 'test'):
            filename = f'{name}_{partition}.json'
            data = spec['extract'](archive, partition)
            # Validate before replacing an installed dataset file.
            validate_records(json.loads(data), partition, spec['label'])
            temporary = folder / (filename + '.tmp')
            temporary.write_bytes(data)
            temporary.replace(folder / filename)
    return describe(root, name=name)


def validate_records(records, partition, label='StrategyQA'):
    if not isinstance(records, list) or not records:
        raise ValueError(f'{label} partition must be a nonempty JSON list')
    ids = set()
    for row in records:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                            for key in ('qid', 'question')):
            raise ValueError(f'{label} rows require nonempty qid and question strings')
        if row['qid'] in ids:
            raise ValueError(f'Duplicate {label} question ID: ' + row['qid'])
        ids.add(row['qid'])
        if 'answer' in row and type(row['answer']) is not bool:
            raise ValueError(f'{label} answer labels must be Boolean')
        if partition == 'train' and 'answer' not in row:
            raise ValueError(f'The labeled {label} train pool requires an answer for every question')


def load(root, spec=None):
    spec = dict(spec or {})
    name = spec.get('name', 'strategyqa')
    pinned = source(name)
    label = pinned['label']
    partition = spec.get('partition', 'train')
    if partition not in ('train', 'test'):
        raise ValueError(f'{label} source partition must be train or test')
    folder = Path(spec.get('path', f'data/{name}'))
    if not folder.is_absolute():
        folder = Path(root) / folder
    file = folder / f'{name}_{partition}.json'
    if not file.is_file():
        raise ValueError(f'{label} is not installed. Run: python -m llm_mas dataset download --name {name}')
    data = file.read_bytes()
    if spec.get('source_sha256') and spec['source_sha256'] != sha(data):
        raise ValueError(f'{label} file changed since the saved configuration')
    rows = json.loads(data)
    validate_records(rows, partition, label)
    metadata = dict(name=name, path=str(folder.relative_to(root)) if folder.is_relative_to(root) else str(folder),
                    partition=partition, source_url=pinned['source_url'], release_sha256=pinned['release_sha256'],
                    source_sha256=sha(data), release_verified=sha(data) == pinned['partition_sha256'][partition],
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
    tasks = [question_task(row['qid'], row['question'], metadata['name']) for row in selected]
    return tasks, metadata, assignments


def labels(root, spec):
    # Evaluator-only entry point. Neither engine nor prompt builders call this.
    rows, _ = load(root, spec)
    return {row['qid']: 'Yes' if row['answer'] else 'No' for row in rows if 'answer' in row}


def describe(root, partition='train', split='train', split_seed=42, name='strategyqa'):
    rows, metadata = load(root, dict(name=name, partition=partition))
    assignments = split_assignments(rows, split_seed, partition)
    return dict(metadata=metadata, labeled=sum('answer' in r for r in rows), total=len(rows),
                splits={s: sum(value == s for value in assignments.values()) for s in sorted(set(assignments.values()))},
                questions=[dict(id=r['qid'], question=r['question']) for r in rows if split == 'all' or assignments[r['qid']] == split])
