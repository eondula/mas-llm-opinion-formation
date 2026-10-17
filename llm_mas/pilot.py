"""Versioned pilot specification, protocol freeze and shared initial-state provenance."""
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from .config import ROOT,digest
from .storage import read,write

PROTOCOL='strategyqa-pilot-v1'
SELECTION='docs/pilot_question_selection.json'
SPEC='studies/pilot_experiment.json'
LOCK='studies/pilot_protocol.lock.json'
FIELDS=('agent_configuration','mode','models','agents','model_configurations','model_revisions','expected_dtype','dataset','question_ids','question_count','question_balance','question_seed','initialization','rounds','repetitions','schedule','conditions','paired_control','decoding','sampling_method','sampling','seed','max_argument_tokens','device','sentiment','message_similarity','stopping','pilot_protocol','stance_audit')


def preset(root=ROOT):
    selection=read(Path(root)/SELECTION)
    config=read(Path(root)/SPEC)
    if config['question_ids']!=selection['question_ids'] or config['dataset']!=dict(selection['source'],split='train',split_seed=selection['split_seed']):
        raise ValueError('Pilot specification differs from frozen question manifest')
    return config


def validate(config,root=ROOT):
    expected=preset(root)
    for field in FIELDS:
        if field not in expected:continue
        actual=config.get(field)
        desired=expected[field]
        if field=='sampling':actual={k:actual.get(k) for k in desired} if isinstance(actual,dict) else actual
        if field=='dataset':actual={k:actual.get(k) for k in desired} if isinstance(actual,dict) else actual
        if actual!=desired:raise ValueError('Frozen pilot setting differs: '+field)
    if config.get('grid') or config.get('agent_counts') or config.get('research') or config.get('observe_stopping'):
        raise ValueError('Pilot does not permit extra factors, research interventions or adaptive stopping')
    selection=read(Path(root)/SELECTION)
    from .datasets import load,split_assignments
    records,_=load(root,config['dataset']);lookup={r['qid']:r for r in records}
    ids=config['question_ids']
    checksum=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    if checksum!=selection['selected_ids_sha256']:raise ValueError('Pilot question selection checksum differs')
    assignments=split_assignments(records,42)
    if len(ids)!=10 or sum(lookup[q]['answer'] for q in ids)!=5 or any(assignments[q]!='train' for q in ids) or set(ids)&set(selection['excluded_question_ids']):
        raise ValueError('Pilot requires the frozen balanced training selection')


def descriptor(config,texts,root=ROOT):
    from .provenance import SCORER_VERSION
    import importlib.metadata
    packages={name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','matplotlib')}
    scientific={k:config.get(k) for k in FIELDS}
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((Path(root)/'llm_mas').glob('*.py'))}
    return dict(protocol=PROTOCOL,settings=scientific,prompts={k:v for k,v in texts.items() if k.startswith('qa_')},source_sha256=sources,
                selection_sha256=hashlib.sha256((Path(root)/SELECTION).read_bytes()).hexdigest(),
                packages=packages,scorer=SCORER_VERSION,measurement_stages=['before_peer_delivery','after_peer_delivery','after_public_response'])


def freeze(root=ROOT):
    from .config import resolve
    from .prompts import templates
    config=resolve(preset(root));value=descriptor(config,templates(),root)
    lock=dict(protocol_hash=digest(value),descriptor=value)
    path=Path(root)/LOCK
    if path.exists() and read(path)!=lock:raise ValueError('Protocol already frozen with different inputs; version the protocol before changing collection settings')
    write(path,lock)
    return lock


def assert_lock(config,texts,root=ROOT):
    from .config import resolve
    validate(config,root)
    path=Path(root)/LOCK
    if not path.exists():raise ValueError('Pilot protocol is not frozen; run pilot-freeze')
    lock=read(path)
    # Per-conversation execution varies only in declared coordinates.
    baseline=resolve(preset(root))
    if digest(descriptor(baseline,texts,root))!=lock['protocol_hash']:raise ValueError('Pilot source, prompts or settings changed after freeze; do not mix protocol versions')
    return lock


def assert_task(active,base):
    for field in FIELDS:
        if field in ('models','model_configurations','schedule','conditions'):continue
        if active.get(field)!=base.get(field):raise ValueError('Pilot task changed frozen setting: '+field)
    if active['models'] not in base['model_configurations'] or active['schedule'] not in ('synchronous','no_peer'):
        raise ValueError('Task is outside frozen pilot composition/branch matrix')


def runtime_options(config,name):
    spec=config.get('model_revisions',{}).get(name)
    if not spec:return {}
    if any(not re.fullmatch('[0-9a-f]{40}',spec.get(k,'')) for k in ('model','tokenizer')):
        raise ValueError('Pin immutable model and tokenizer commit IDs')
    return dict(revision=spec['model'],tokenizer_revision=spec['tokenizer'],expected_dtype=config.get('expected_dtype'))


def shared_initialization(path,config,task,texts,runtimes,protocol_hash,progress=lambda **e:None):
    from .engine import conversation
    coordinates=dict(question_id=task['id'],models=config['models'],agents=config['agents'],repetition=config.get('repetition',0),
                     seed_offset=config.get('seed_offset',0),sampling=config.get('sampling'),initialization=config['initialization'],protocol_hash=protocol_hash)
    pair_id=digest(coordinates)[:24]
    file=Path(path)/'initializations'/f'{pair_id}.json'
    if file.exists():
        snapshot=read(file)
        if snapshot['coordinates']!=coordinates or digest(snapshot['rows'])!=snapshot['initialization_hash']:raise ValueError('Shared initialization snapshot changed')
    else:
        rows=conversation(dict(config,rounds=0,schedule='synchronous'),task,texts,runtimes,emit=progress)
        snapshot=dict(pair_id=pair_id,coordinates=coordinates,rows=rows,initialization_hash=digest(rows))
        write(file,snapshot)
    return deepcopy(snapshot)
