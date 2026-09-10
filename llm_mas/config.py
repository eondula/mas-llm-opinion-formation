import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def scenario(value):
    path = Path(value)
    if not path.is_file():
        path = ROOT / 'scenarios' / 'controlled' / f'{value}.json'
    data = json.loads(path.read_text())
    validate_scenario(data)
    return data

def validate_scenario(data):
    for field in ('id', 'title'):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f'Missing scenario {field}')
    for key in ('strategy_1', 'strategy_2'):
        s = data.get(key, {})
        for field in ('name', 'property_1', 'property_2', 'candidate'):
            if not isinstance(s.get(field), str) or not s[field].strip():
                raise ValueError(f'Missing {key}.{field}')
        if not isinstance(s.get('arguments'), list) or not s['arguments'] or not all(isinstance(x, str) and x.strip() for x in s['arguments']):
            raise ValueError(f'Invalid arguments for {key}')
    for pair in data.get('candidate_variants', []):
        if not isinstance(pair,list) or len(pair)!=2 or not all(isinstance(v,str) and v.strip() for v in pair): raise ValueError('Candidate variants require pairs of nonempty strings')
    if data['strategy_1']['candidate'] == data['strategy_2']['candidate']:
        raise ValueError('Candidate statements must differ')

def resolve(raw):
    c = dict(mode='preview', models=[DEFAULT_MODEL], agents=2, scenarios=['storage'], rounds=20,
             initialization='natural', schedule='synchronous', argument_mode='predefined',
             max_argument_tokens=128, sentiment=True, device='auto', presentation_order='strategy_1_first')
    c.update(raw)
    c['conditions'] = c.get('conditions') or []
    if c.pop('paper_suite', False):
        import json
        paper=json.loads((ROOT/'studies/paper.json').read_text())
        c['research']=paper['research']
        c['grid']=paper['grid']
        c['conditions']=[]
        c['research_events_per_conversation']=1
        enriched=[]
        for value in c['scenarios']:
            name=Path(value).stem if isinstance(value,str) else ''
            candidate=ROOT/'scenarios/study'/f'{name}.json'
            enriched.append(str(candidate) if name in ('sensor','storage') and candidate.exists() else value)
        c['scenarios']=enriched
    if c['mode'] not in ('preview', 'study'): raise ValueError('Invalid mode')
    if not isinstance(c['agents'], int) or c['agents'] < 2: raise ValueError('At least two agents required')
    if not isinstance(c['rounds'], int) or c['rounds'] < 1: raise ValueError('Rounds must be positive')
    if c['mode'] == 'preview' and (c['agents'] != 2 or len(c['models']) != 1 or len(c['scenarios']) != 1):
        raise ValueError('Preview requires two agents, one model and one scenario')
    if len(c['models']) not in (1, c['agents']) or not all(isinstance(m, str) and m.strip() for m in c['models']):
        raise ValueError('Supply one model or one checkpoint per agent')
    if c['schedule'] not in ('synchronous', 'sequential', 'reverse_sequential', 'no_peer'): raise ValueError('Invalid schedule')
    if c['argument_mode'] not in ('predefined', 'generated'): raise ValueError('Invalid argument mode')
    if c['presentation_order'] not in ('strategy_1_first', 'strategy_2_first'): raise ValueError('Invalid presentation order')
    if c['device'] not in ('auto', 'cpu', 'cuda', 'mps'): raise ValueError('Invalid device')
    if not isinstance(c['max_argument_tokens'], int) or not 1 <= c['max_argument_tokens'] <= 2048: raise ValueError('Invalid token limit')
    start = c['initialization']
    if start != 'natural' and (not isinstance(start, list) or len(start) != c['agents'] or any(x not in (1,2) for x in start)):
        raise ValueError('Initialization must be natural or a list of 1/2, one per agent')
    records = [scenario(x) if isinstance(x, str) else x for x in c['scenarios']]
    for record in records: validate_scenario(record)
    if not records or len({x['id'] for x in records}) != len(records): raise ValueError('Scenarios must be nonempty and have unique IDs')
    if set(c.get('research', [])) - {'validation', 'history', 'replay', 'prediction', 'mechanistic', 'heads', 'source_spans', 'continuation', 'sufficiency'}: raise ValueError('Unknown research module')
    if c['mode'] == 'preview' and (c.get('conditions') or c.get('research')): raise ValueError('Preview runs one conversation; conditions and research require study mode')
    for condition in c.get('conditions', []):
        if set(condition) - {'initialization','schedule','argument_mode','presentation_order'}: raise ValueError('Invalid condition fields')
        resolve(dict(c, **condition, conditions=[]))
    c['scenarios'] = records
    return c

def run_path(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name): raise ValueError('Run ID must contain letters, digits, underscores or hyphens')
    return ROOT / 'results' / 'studies' / name


def resolve_run_id(value=None):
    from datetime import datetime
    name = value.strip() if isinstance(value, str) else value
    if not name:
        name = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    run_path(name)
    return name
