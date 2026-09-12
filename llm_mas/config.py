import math
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
        candidates=[ROOT/path,ROOT/'scenarios/controlled'/f'{value}.json',ROOT/'scenarios/custom'/f'{value}.json']
        path=next((p for p in candidates if p.is_file()),candidates[1])
    data = json.loads(path.read_text())
    validate_scenario(data)
    return data

def validate_scenario(data):
    if not isinstance(data,dict):raise ValueError('Scenario must be a JSON object')
    for field in ('id', 'title'):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f'Missing scenario {field}')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',data['id']):raise ValueError('Scenario ID may contain only letters, numbers, underscores and hyphens')
    for key in ('strategy_1', 'strategy_2'):
        s = data.get(key, {})
        if not isinstance(s,dict):raise ValueError(f'{key} must be an object')
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
    if raw.get('mode') == 'no_fixed_rounds':
        raw = dict(raw, mode='preview', stopping='sign_stability')
    if raw.get('preview_preset'):
        if raw['preview_preset'] not in ('persona', 'no_persona'):
            raise ValueError('Unknown quick preview preset')
        raw = dict(raw, persona=raw['preview_preset'] == 'persona')
    if raw.get('mode') == 'experiment_c':
        preset = json.loads((ROOT/'studies/experiment_C.json').read_text())
        # Use protocol defaults, but only expand factors explicitly requested.
        for key in ('grid', 'model_configurations', 'export_profile'):
            preset.pop(key, None)
        raw = dict(preset, **raw)
        raw['mode'] = 'study'
        raw['preset'] = 'experiment_c'


    c = dict(mode='preview', models=[DEFAULT_MODEL], agents=2, scenarios=['storage'], rounds=20,
             initialization='natural', schedule='synchronous', argument_mode='predefined',
             max_argument_tokens=128, sentiment=True, device='auto', presentation_order='strategy_1_first')
    c.update(raw)
    c['decoding'] = c.get('decoding', 'greedy')
    if c['decoding'] not in ('greedy', 'fixed_sampling'):
        raise ValueError('Unknown decoding mode')
    if c['decoding'] == 'fixed_sampling':
        if c['argument_mode'] not in ('generated', 'conversational', 'combined_reasoning'):
            raise ValueError('Sampling requires generated explanations; choose Combined prompt (flexible reasoning) or another generated mode')
        method = c.get('sampling_method', 'combined')
        if method not in ('temperature', 'top_p', 'combined'):
            raise ValueError('Unknown sampling method')
        supplied = c.get('sampling') or {}
        if not isinstance(supplied, dict):
            raise ValueError('Sampling settings must be an object')
        settings = dict(temperature=0.7, top_p=0.9, seed=42)
        settings.update({k:supplied[k] for k in settings if k in supplied})
        if method == 'temperature':settings['top_p']=1.0
        if method == 'top_p':settings['temperature']=1.0
        for key in ('temperature','top_p'):
            value=settings[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError(f'{key} must be a finite positive number')
        if settings['top_p']>1:raise ValueError('top_p must be at most 1')
        if type(settings['seed']) is not int or not 0<=settings['seed']<=2**32-1:
            raise ValueError('Seed must be an integer from 0 to 4294967295')
        settings['top_k']=0
        c['sampling_method']=method
        c['sampling']=settings
    else:
        c.pop('sampling', None)
    c['conditions'] = c.get('conditions') or []
    if c.pop('paper_suite', False):
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
    if c.get('stopping', 'fixed') not in ('fixed', 'exact_repetition', 'cosine_similarity', 'combined_stability', 'sign_stability'):
        raise ValueError('Invalid stopping policy')
    if c.get('stopping') in ('exact_repetition', 'cosine_similarity', 'combined_stability', 'sign_stability') and c.get('export_profile') in ('experiment_c', 'experiment_d'):
        raise ValueError('Adaptive stopping requires the generic figure exporter')
    if c.get('stopping') == 'sign_stability':
        c.pop('minimum_rounds', None)
        c.pop('margin_tolerance', None)
        c.setdefault('similarity_threshold', 0.8)
        threshold = c['similarity_threshold']
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError('Similarity threshold must be between -1 and 1')
    if c.get('stopping') == 'combined_stability':
        c.pop('minimum_rounds', None)
        c.setdefault('margin_tolerance', 0.1)
        tolerance = c['margin_tolerance']
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError('Margin tolerance must be finite and nonnegative')
        c.setdefault('similarity_threshold', 0.8)
        threshold = c['similarity_threshold']
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError('Similarity threshold must be between -1 and 1')
    if c.get('stopping') == 'cosine_similarity':
        c.setdefault('minimum_rounds', 2)
        if type(c['minimum_rounds']) is not int or c['minimum_rounds'] < 1:
            raise ValueError('Minimum rounds must be a positive integer')
        c.setdefault('similarity_threshold', 0.8)
        threshold = c['similarity_threshold']
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError('Similarity threshold must be between -1 and 1')
    if c['mode'] not in ('preview', 'study'): raise ValueError('Invalid mode')
    if not isinstance(c['agents'], int) or c['agents'] < 2: raise ValueError('At least two agents required')
    if not isinstance(c['rounds'], int) or c['rounds'] < 1: raise ValueError('Rounds must be positive')
    if c['mode'] == 'preview' and len(c['scenarios']) != 1:
        raise ValueError('Preview requires one scenario')
    if len(c['models']) not in (1, c['agents']) or not all(isinstance(m, str) and m.strip() for m in c['models']):
        raise ValueError('Supply one model or one checkpoint per agent')
    if c.get('agent_configuration') == 'homogeneous' and len(set(c['models'])) != 1:
        raise ValueError('Homogeneous mode requires a shared checkpoint')
    if c.get('agent_configuration') == 'heterogeneous' and len(c['models']) != c['agents']:
        raise ValueError('Per-agent mode requires one checkpoint selection per agent; repeated checkpoints are allowed')
    if c['mode']=='preview' and any(c.get(k) for k in ('grid','agent_counts','model_configurations')):
        raise ValueError('Preview runs one conversation; sweeps require study mode')
    if c['schedule'] not in ('synchronous', 'sequential', 'reverse_sequential', 'no_peer'): raise ValueError('Invalid schedule')
    if c['argument_mode'] not in ('predefined', 'agent_selected', 'generated', 'conversational', 'combined_fixed', 'combined_reasoning'): raise ValueError('Invalid argument mode')
    if c['presentation_order'] not in ('strategy_1_first', 'strategy_2_first'): raise ValueError('Invalid presentation order')
    if c['device'] not in ('auto', 'cpu', 'cuda', 'mps'): raise ValueError('Invalid device')
    if not isinstance(c['max_argument_tokens'], int) or not 1 <= c['max_argument_tokens'] <= 2048: raise ValueError('Invalid token limit')
    if c.get('decision_mode', 'scored') not in ('scored','generated'):
        raise ValueError('Invalid decision mode')
    if c.get('decision_mode') == 'generated' and c['argument_mode'] != 'combined_reasoning':
        raise ValueError('Generated choice requires combined reasoning')
    if type(c.get('observe_stopping',False)) is not bool:
        raise ValueError('observe_stopping must be boolean')
    start = c['initialization']
    if start != 'natural' and (not isinstance(start, list) or len(start) != c['agents'] or any(x not in (1,2) for x in start)):
        raise ValueError('Initialization must be natural or a list of 1/2, one per agent')
    records = [scenario(x) if isinstance(x, str) else x for x in c['scenarios']]
    for record in records: validate_scenario(record)
    if not records or len({x['id'] for x in records}) != len(records): raise ValueError('Scenarios must be nonempty and have unique IDs')
    if set(c.get('research', [])) - {'validation', 'history', 'replay', 'prediction', 'mechanistic', 'heads', 'source_spans', 'continuation', 'sufficiency'}: raise ValueError('Unknown research module')
    if c['mode'] == 'preview' and (c.get('conditions') or c.get('research')): raise ValueError('Preview runs one conversation; conditions and research require study mode')
    for condition in c.get('conditions', []):
        if set(condition) - {'initialization','schedule','argument_mode','presentation_order','persona','decision_mode','stopping','observe_stopping'}: raise ValueError('Invalid condition fields')
        resolve(dict(c, **condition, conditions=[]))
    if not isinstance(c.get('persona', False), bool):
        raise ValueError('Persona must be true or false')
    if c.get('persona'):
        for record in records:
            roles = record.get('persona_roles', {})
            for key in ('strategy_1', 'strategy_2'):
                role = roles.get(key, {})
                if not all(isinstance(role.get(f), str) and role[f].strip() for f in ('role_name', 'priority')):
                    raise ValueError('Persona requires scenario role_name and priority for both strategies')
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
