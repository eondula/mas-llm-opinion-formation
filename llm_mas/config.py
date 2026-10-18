import math
import hashlib
import json
import re
from pathlib import Path
from .locations import input_path as locate_input, study_roots

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'


def input_path(value):
    return locate_input(value, ROOT)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def scenario(value):
    path = input_path(value)
    if not path.is_file():
        candidates=[path,input_path(Path('scenarios/controlled')/f'{value}.json'),input_path(Path('scenarios/custom')/f'{value}.json')]
        path=next((p for p in candidates if p.is_file()),candidates[1])
    data = json.loads(path.read_text())
    validate_scenario(data)
    return data

def validate_scenario(data):
    if not isinstance(data,dict):raise ValueError('Scenario must be a JSON object')
    from .tasks import is_question, validate_question
    if is_question(data):
        return validate_question(data)
    for field in ('id', 'title'):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f'Missing scenario {field}')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',data['id']):raise ValueError('Scenario ID may contain only letters, numbers, underscores and hyphens')
    if 'context' in data and (not isinstance(data['context'], str) or not data['context'].strip()):
        raise ValueError('Scenario context must be nonempty text')
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

def resolve(raw, planned=False):
    if raw.get('dataset') or (raw.get('mode') == 'experiment' and 'scenarios' not in raw):
        from .datasets import select
        if raw.get('mode', 'experiment') not in ('experiment', 'preview', 'study'):
            raise ValueError('StrategyQA uses the unified experiment protocol, not historical joint/exploration modes')
        tasks, dataset, _ = select(ROOT, raw)
        required = dict(decision_mode='generated',
                        argument_mode='combined_reasoning', prompt_style='question', stopping='fixed')
        for key, value in required.items():
            if key in raw and raw[key] != value:
                raise ValueError(f'StrategyQA requires {key}={value}')
        if (raw.get('persona') or raw.get('preview_preset') == 'persona' or raw.get('paper_suite')
                or raw.get('research') or raw.get('export_profile') or raw.get('joint_task_brief')
                or raw.get('argument_source', 'own') != 'own' or raw.get('information_access', 'none') != 'none'):
            raise ValueError('Historical personas, argument probes, joint tasks and exporters are not StrategyQA protocols')
        raw=dict(raw)
        raw.setdefault('response_protocol','legacy_json' if raw.get('pilot_protocol','').startswith('strategyqa-pilot-v4-') else 'reasoning_answer_v3')
        if raw['response_protocol']=='reasoning_answer_v3':raw.setdefault('copy_block_ngram',6)
        raw.setdefault('question_figure_profile','question_png')
        raw.setdefault('max_argument_tokens',512)
        raw = dict(raw, **required, dataset=dataset, tasks=tasks, scenarios=tasks,
                   question_ids=[t['id'] for t in tasks], question_count=len(tasks))
    if raw.get('mode') == 'scenario_exploration':
        for key, expected in {'decision_mode':'generated','argument_mode':'combined_reasoning','prompt_style':'original','stopping':'fixed','decoding':'greedy','presentation_order':'strategy_1_first'}.items():
            if key in raw and raw[key] != expected:
                raise ValueError('Exploration fixes '+key+' to '+expected)
        if any(raw.get(k) for k in ('joint_task_brief','observe_stopping','reverse_candidate_blocks','export_profile','paper_suite','sampling','conditions','research','grid','agent_counts','model_configurations')):
            raise ValueError('Exploration does not accept closure, study, sampling or reversal options')
        raw = dict(raw, decision_mode='generated', argument_mode='combined_reasoning', prompt_style='original', stopping='fixed', decoding='greedy', presentation_order='strategy_1_first')
        raw.setdefault('max_argument_tokens', 256)
        raw.setdefault('rounds', 5)
        raw.setdefault('argument_source', 'own')
        raw.setdefault('stance_diagnostic', False)
        if raw['argument_source'] not in ('own','supplied') or type(raw['stance_diagnostic']) is not bool:
            raise ValueError('Invalid exploration source or diagnostic setting')
    if raw.get('mode') == 'no_fixed_rounds':
        raw = dict(raw, mode='preview', stopping='sign_stability')
    if raw.get('preview_preset'):
        if raw['preview_preset'] not in ('persona', 'no_persona'):
            raise ValueError('Unknown quick preview preset')
        raw = dict(raw, persona=raw['preview_preset'] == 'persona')
    if raw.get('mode') == 'experiment_c':
        preset = json.loads(input_path('studies/experiment_C.json').read_text())
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
    if c.get('output_layout') not in (None, 'experiment'):
        raise ValueError('Unknown output layout')
    if c.get('dataset'):
        if c.get('response_protocol') not in ('legacy_json','decision_time_v2','reasoning_answer_v3'):raise ValueError('Unknown question response protocol')
        if c.get('question_figure_profile') not in ('question_png','full'):raise ValueError('Unknown question figure profile')
        if c['response_protocol'] in ('decision_time_v2','reasoning_answer_v3') and c['max_argument_tokens']<2:raise ValueError('Decision-time response budget must be at least two tokens')
        if c['response_protocol']=='reasoning_answer_v3':
            n=c.get('copy_block_ngram')
            if type(n) is not int or not (n==0 or 2<=n<=32):raise ValueError('copy_block_ngram must be 0 (off) or an integer from 2 to 32')
        elif 'copy_block_ngram' in c:raise ValueError('copy_block_ngram applies only to reasoning_answer_v3')
        c.setdefault('message_similarity', True)
        if type(c['message_similarity']) is not bool:
            raise ValueError('message_similarity must be boolean')
    c.setdefault('repetitions', 1)
    c.setdefault('seed', 42)
    c.setdefault('stopping', 'fixed')
    if type(c['repetitions']) is not int or c['repetitions'] < 1:
        raise ValueError('Repetitions must be a positive integer')
    if type(c['seed']) is not int or not 0 <= c['seed'] <= 2**32 - 1:
        raise ValueError('Seed must be an integer from 0 to 4294967295')
    if c['mode'] == 'joint_task':
        from .joint_task import DEFAULT_BRIEF
        c.setdefault('joint_task_brief', DEFAULT_BRIEF)
        if not isinstance(c['joint_task_brief'], str) or not c['joint_task_brief'].strip() or len(c['joint_task_brief']) > 3000:
            raise ValueError('Joint task brief must contain 1–3000 characters')
    c.setdefault('prompt_style', 'original')
    if c['prompt_style'] not in ('original', 'open_discussion', 'topic_discussion', 'discussion_then_decision', 'question'):
        raise ValueError('Invalid prompt style')
    if c['prompt_style'] != 'original' and c['argument_mode'] != 'combined_reasoning':
        raise ValueError('Discussion prompt presets require combined_reasoning')
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
        settings = dict(temperature=0.7, top_p=0.9, seed=c['seed'])
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
        paper=json.loads(input_path('studies/paper.json').read_text())
        c['research']=paper['research']
        c['grid']=paper['grid']
        c['conditions']=[]
        c['research_events_per_conversation']=1
        enriched=[]
        for value in c['scenarios']:
            name=Path(value).stem if isinstance(value,str) else ''
            candidate=input_path(Path('scenarios/study')/f'{name}.json')
            enriched.append(str(candidate) if name in ('sensor','storage') and candidate.exists() else value)
        c['scenarios']=enriched
    if c.get('stopping', 'fixed') not in ('fixed', 'exact_repetition', 'cosine_similarity', 'combined_stability', 'sign_stability', 'agent_closure'):
        raise ValueError('Invalid stopping policy')
    if c.get('stopping') in ('exact_repetition', 'cosine_similarity', 'combined_stability', 'sign_stability') and c.get('export_profile') in ('experiment_c', 'experiment_d'):
        raise ValueError('Adaptive stopping requires the generic figure exporter')
    if c.get('stopping') == 'agent_closure' and (c.get('observe_stopping') or c.get('export_profile')):
        raise ValueError('Agent closure requires normal stopping and the generic exporter')
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
    if c['mode'] not in ('experiment', 'preview', 'study', 'joint_task', 'scenario_exploration'): raise ValueError('Invalid mode')
    if type(c['agents']) is not int or c['agents'] < 1: raise ValueError('At least one agent required')
    if type(c['rounds']) is not int or c['rounds'] < 0: raise ValueError('Update rounds must be nonnegative')
    if c['rounds'] == 0 and c['stopping'] != 'fixed':
        raise ValueError('Adaptive stopping requires at least one update round')
    if c['mode'] in ('preview', 'joint_task', 'scenario_exploration') and len(c['scenarios']) != 1:
        raise ValueError('Preview requires one scenario')
    if len(c['models']) not in (1, c['agents']) or not all(isinstance(m, str) and m.strip() for m in c['models']):
        raise ValueError('Supply one model or one checkpoint per agent')
    assignments = c.get('model_configurations')
    if assignments:
        if not isinstance(assignments, list) or any(
            not isinstance(row, list) or not any(len(row) in (1, n) for n in (c.get('agent_counts') or [c['agents']]))
            or not all(isinstance(model, str) and model.strip() for model in row)
            for row in assignments
        ):
            raise ValueError('Each model composition requires one checkpoint or one checkpoint per agent')
        expanded = [tuple(row * c['agents'] if len(row) == 1 else row) for row in assignments]
        if len(set(expanded)) != len(expanded):
            raise ValueError('Model compositions must be unique')
    if c.get('agent_configuration') == 'homogeneous' and len(set(c['models'])) != 1:
        raise ValueError('Homogeneous mode requires a shared checkpoint')
    if c.get('agent_configuration') == 'heterogeneous' and len(c['models']) != c['agents']:
        raise ValueError('Per-agent mode requires one checkpoint selection per agent; repeated checkpoints are allowed')
    if c['mode'] in ('preview', 'joint_task', 'scenario_exploration') and any(c.get(k) for k in ('grid','agent_counts','model_configurations')):
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
    if c['mode'] in ('preview', 'joint_task', 'scenario_exploration') and (c.get('conditions') or c.get('research')): raise ValueError('Preview runs one conversation; conditions and research require study mode')
    for condition in c.get('conditions', []):
        if set(condition) - {'initialization','schedule','argument_mode','presentation_order','persona','decision_mode','stopping','observe_stopping','prompt_style'}: raise ValueError('Invalid condition fields')
        resolve(dict(c, **condition, conditions=[]),planned=True)
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
    if c['mode'] == 'scenario_exploration':
        c.setdefault('exploration_setup', 'standard')
        if c['exploration_setup'] not in ('standard', 'deliberation'):
            raise ValueError('Invalid exploration setup')
        if c['exploration_setup'] == 'deliberation':
            if c['argument_source'] != 'own' or c.get('information_access') not in ('shared', 'distributed'):
                raise ValueError('Deliberation requires own arguments and shared or distributed evidence')
            if any(s.get('scenario_family') != 'water_monitoring' for s in c['scenarios']):
                raise ValueError('Deliberation requires a water monitoring family scenario')
        c.setdefault('information_access', 'none')
        c.setdefault('evidence_rotation', 0)
        c.setdefault('persona_rotation', 0)
        if c['information_access'] not in ('none', 'shared', 'distributed'):
            raise ValueError('Invalid information access')
        for key in ('evidence_rotation', 'persona_rotation'):
            if type(c[key]) is not int or c[key] < 0:
                raise ValueError(key+' must be a nonnegative integer')
        if c['information_access'] != 'none':
            if c['argument_source'] != 'own':
                raise ValueError('Evidence conditions require own arguments to avoid leaking private evidence through argument lists')
            for evidence_scenario in c['scenarios']:
                cards = evidence_scenario.get('evidence_cards', [])
                if not cards or any(not isinstance(card, dict) or not isinstance(card.get('id'), str) or not card['id'] or not isinstance(card.get('text'), str) or not card['text'].strip() for card in cards):
                    raise ValueError('Select a scenario with valid evidence cards')
                if len({card['id'] for card in cards}) != len(cards):
                    raise ValueError('Evidence IDs must be unique')
    if type(c.get('paired_control',False)) is not bool:raise ValueError('paired_control must be boolean')
    if c.get('paired_control') and not planned:
        if not c.get('dataset') or c['stopping']!='fixed' or len(c['conditions'])!=2 or {v.get('schedule') for v in c['conditions']}!={c['schedule'],'no_peer'} or c['schedule']=='no_peer':
            raise ValueError('Paired controls require one peer branch and one no-peer branch with fixed horizon')
    if c.get('pilot_protocol'):
        from .pilot import paths,validate
        paths(c['pilot_protocol'])
        if not planned:validate(c)
    return c

def run_path(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name): raise ValueError('Run ID must contain letters, digits, underscores or hyphens')
    experiment = ROOT / 'results/experiments' / name
    if experiment.exists():
        return experiment
    active, archived = (parent / name for parent in study_roots(ROOT))
    return archived if not active.exists() and archived.exists() else active


def resolve_run_id(value=None):
    from datetime import datetime
    name = value.strip() if isinstance(value, str) else value
    if not name:
        name = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    run_path(name)
    return name
