import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from .config import resolve, digest, run_path, ROOT, resolve_run_id
from .prompts import templates
from .engine import conversation


from .storage import write, read  # Keep write available for existing callers.
from .provenance import source_hashes, environment
from .decision_generation import DECISION_PROTOCOLS
from .trajectories import SCHEMA_VERSION, read_conversation


def execute(raw, name=None, resume=False, launch_token=None):
    config = resolve(raw)
    configured_name=config.pop('run_id',None)
    name = resolve_run_id(name or configured_name)
    path = ROOT/'results/experiments'/name if config.get('output_layout') == 'experiment' else run_path(name)
    existing = run_path(name)
    if existing != path and existing.exists():
        raise ValueError('Experiment ID is already used by another saved run')
    texts = templates()
    dataset_snapshot = None
    if config.get('dataset'):
        from .datasets import select
        selected, metadata, assignments = select(ROOT, config)
        dataset_snapshot = dict(metadata=metadata, selected_question_ids=[t['id'] for t in selected],
                                assignments=assignments)
    from .planning import configurations
    tasks = [resolve(c,planned=True) for c in configurations(config)]
    if not tasks: raise ValueError('No compatible model/agent configurations')
    identity = digest(dict(config=config, prompts=texts))
    protocol_hash=identity
    if config.get('pilot_protocol'):
        from .pilot import assert_lock
        protocol_hash=assert_lock(config,texts)['protocol_hash']
    if launch_token:
        if resume:raise ValueError('Dashboard launch token cannot be used to resume existing data')
        from .launch import claim
        reserved=claim(path,config,launch_token)
    else:
        # Backward compatibility for older launchers; completed scientific data is never reused.
        from .launch import legacy_reservation
        reserved = config.get('output_layout') == 'experiment' and legacy_reservation(path,config)
    if path.exists() and not reserved:
        if not resume: raise ValueError(f'Run already exists: {path}')
        manifest = json.loads((path/'manifest.json').read_text())
        if manifest.get('source_hashes') != source_hashes(): raise ValueError('Source code changed since run; start a new run instead of resuming')
        if manifest['input_hash'] != identity: raise ValueError('Resume inputs differ from saved configuration/prompts')
    else:
        path.mkdir(parents=True, exist_ok=reserved)
        write(path/'config.json', config)
        write(path/'inputs/prompts.json', texts)
        write(path/'inputs/scenarios.json', config['scenarios'])
        if dataset_snapshot:
            write(path/'inputs/tasks.json', config['tasks'])
            write(path/'inputs/dataset.json', dataset_snapshot)
        write(path/'manifest.json', dict(input_hash=identity, schema_version=2, created=datetime.now().isoformat(),
                                       resume_unit='completed conversation', source_hashes=source_hashes()))
    if config.get('pilot_protocol'):
        write(path/'inputs/protocol.json',assert_lock(config,texts))
    start = time.monotonic()
    write(path/'tasks.json', tasks)
    total = sum(len(c['scenarios'])*(c['rounds']+1)*c['agents'] for c in tasks)
    done = 0
    generated_done=0
    generation_total=total
    if config.get('paired_control'):
        blocks={(tuple(c['models']),c['agents'],c.get('repetition',0),si) for c in tasks for si,_ in enumerate(c['scenarios'])}
        generation_total=total-sum(c['agents']*len(c['scenarios']) for c in tasks)+sum(b[1] for b in blocks)
        generated_done=sum(len(read(f)['rows']) for f in (path/'initializations').glob('*.json'))
        generated_done+=sum(max(0,len(read(f)['rows'])-read(f)['config']['agents']) for f in (path/'conversations').glob('*.json'))
    runtimes = {}
    previous_environment = read(path/'environment.json') if (path/'environment.json').exists() else {}
    def save_environment():
        metadata = environment(config, runtimes)
        metadata['models'] = dict(previous_environment.get('models', {}), **metadata['models'])
        write(path/'environment.json', metadata)
    save_environment()
    from .progress import Progress
    tracker=Progress()
    phase_counts={}
    termination_status = {}
    def status(phase, **extra):
        metrics=tracker.update(phase,extra.pop('phase_completed',done if phase=='conversation' else None),extra.pop('phase_total',total if phase=='conversation' else None))
        write(path/'status.json', dict(metrics, completed=done,total=total,generated_responses=generated_done,generation_total=generation_total,termination=termination_status or None,**extra))
    def research_progress(module,model,scenario):
        phase_counts[module]=phase_counts.get(module,0)+1
        if module in ('validation','history'):
            phase_total=len(runtimes)*len(config['scenarios'])
        else:
            phase_total=sum(min(config.get('research_events_per_conversation',2),sum(r['round']>0 and bool(r.get('deliveries')) for r in json.loads(f.read_text())['rows'])) for f in (path/'conversations').glob('*.json'))
        status(module,phase_completed=phase_counts[module],phase_total=phase_total,model=model,scenario=scenario)
        print(f"{module}: {phase_counts[module]} cases completed · elapsed {time.monotonic()-start:.0f}s",flush=True)
    status('loading')
    try:
        from .runtime import Runtime
        from .pilot import runtime_options
        def load_runtime(name_,active):
            if name_ not in runtimes:
                status('loading',model=name_)
                runtimes[name_]=Runtime(name_,active['device'],**runtime_options(active,name_))
                save_environment()
        if config.get('stance_audit'):
            for model_name in config.get('model_revisions',{}):load_runtime(model_name,config)
            from .stance_audit import audit
            status('stance_audit')
            audit(path,config['tasks'],texts,runtimes,protocol_hash,lambda a,b:status('stance_audit',phase_completed=a,phase_total=b),make_figures=config.get('question_figure_profile')!='question_png',measurement_role='auxiliary_probe' if config.get('response_protocol') in DECISION_PROTOCOLS else 'primary_probe')
        for ti, active in enumerate(tasks):
            for si, scenario in enumerate(active['scenarios']):
                target = path/'conversations'/f'{ti:03d}_{si:03d}.json'
                if target.exists():
                    saved = read_conversation(target)
                    termination_status = saved.get('termination') or {}
                    count = len(saved['rows'])
                    done += count
                    if not config.get('paired_control'):generated_done+=count
                    total -= (active['rounds']+1)*active['agents'] - count
                    generation_total -= (active['rounds']+1)*active['agents'] - count
                    continue
                if config.get('pilot_protocol'):
                    assert_lock(config,texts)
                    from .pilot import assert_task
                    assert_task(active,config)
                for name_ in set(active['models']):load_runtime(name_,active)
                base = done
                def emit(**event):
                    nonlocal done,generated_done
                    done = base+event['completed']
                    if event.get('kind')=='response' and not event.get('generation_reused'):generated_done+=1
                    with (path/'events.jsonl').open('a') as out:
                        out.write(json.dumps(dict(task=target.stem, **event))+'\n')
                    status('stopping_check' if event.get('kind') == 'termination_check' else 'conversation', task=target.stem)
                    elapsed = time.monotonic()-start
                    print(f'[{"="*int(30*done/total):30}] {done}/{total} elapsed {elapsed:.0f}s ETA {elapsed*(total-done)/done:.0f}s', flush=True)
                execution = dict(active)
                if active.get('repetitions', 1) > 1:
                    execution['seed_offset'] += si * (active['rounds'] + 1) * active['agents']
                if config.get('paired_control') and active['schedule'] == 'no_peer' and active.get('stopping') == 'agent_signal':
                    # The matched peer conversation runs first; its stop round sets the control's length.
                    partner = next((pi for pi, c in enumerate(tasks) if c['schedule'] != 'no_peer' and c['models'] == active['models']
                                    and c['agents'] == active['agents'] and c.get('repetition', 0) == active.get('repetition', 0)
                                    and c['initialization'] == active['initialization']), None)
                    source = path/'conversations'/f'{partner:03d}_{si:03d}.json' if partner is not None else None
                    if source is None or not source.exists():
                        raise ValueError('The no-peer control needs its matched peer conversation to finish first')
                    execution['yoked_rounds'] = read_conversation(source)['termination']['round']
                snapshot=None
                if config.get('paired_control'):
                    from .pilot import shared_initialization
                    def initial_progress(**event):
                        nonlocal generated_done
                        generated_done+=1
                        # Publish each shared starting response immediately; the
                        # normal branch replay is deduplicated by response identity.
                        with (path/'events.jsonl').open('a') as out:
                            out.write(json.dumps(dict(task=target.stem, **event))+'\n')
                        status('initialization',phase_completed=generated_done,phase_total=generation_total)
                    snapshot=shared_initialization(path,execution,scenario,texts,runtimes,protocol_hash,initial_progress)
                rows = conversation(execution, scenario, texts, runtimes, emit,**({'initial_rows':snapshot['rows']} if snapshot else {}))
                total -= (active['rounds']+1)*active['agents'] - len(rows)
                generation_total -= (active['rounds']+1)*active['agents'] - len(rows)
                termination = rows[-1].get('termination')
                if active.get('observe_stopping'):
                    triggers = [r['round'] for r in rows if (r.get('termination') or {}).get('reason') == 'system_detected_stability']
                    termination = dict(reason='fixed_horizon', round=rows[-1]['round'], first_proposed_stop=min(triggers) if triggers else None)
                termination_status = termination or {}
                record = dict(schema_version=SCHEMA_VERSION, config=execution, scenario=scenario,
                              question_id=scenario['id'], repetition=active.get('repetition', 0),
                              measurement_stages=dict(before='before_peer_delivery',
                                  decision='answer_token_after_reasoning' if config.get('response_protocol')=='reasoning_answer_v3' else 'public_answer_token_before_explanation' if config.get('response_protocol')=='decision_time_v2' else 'after_peer_delivery_and_optional_discussion', post='after_public_response_probe',primary='decision_time' if config.get('response_protocol') in DECISION_PROTOCOLS else 'post'),
                              rows=rows, termination=termination)
                if snapshot:
                    record.update(pair_id=snapshot['pair_id'],initialization_hash=snapshot['initialization_hash'],control_design='shared_initialization',protocol_hash=protocol_hash)
                if config.get('dataset'):
                    record.update(task=scenario, dataset=config['dataset'],
                                  research_split=dataset_snapshot['assignments'][scenario['id']])
                if active['mode'] == 'joint_task':
                    record['joint_outcome'] = rows[-1]['joint_outcome']
                    write(path/'joint_decision.json', dict(task=target.stem, brief=active['joint_task_brief'], **record['joint_outcome']))
                write(target, record)
        if config.get('research'):
            from .research import run as run_research
            for model_name in {m for task in tasks for m in task['models']}:
                if model_name not in runtimes: runtimes[model_name] = Runtime(model_name, config['device'])
            status('research')
            run_research(path, config, texts, runtimes, research_progress)
        save_environment()
        runtimes.clear()
        import gc
        gc.collect()
        if config.get('dataset'):
            status('evaluation')
            from .datasets import labels
            from .evaluation import evaluate
            evaluation_labels = labels(ROOT, config['dataset'])
            write(path/'inputs/evaluation_labels.json', {qid:evaluation_labels[qid] for qid in config['question_ids'] if qid in evaluation_labels})
            evaluate(path, evaluation_labels)
        if config['sentiment']:
            status('sentiment')
            from .sentiment import measure_sentiment
            cache = {}
            sentiment_total=sum(len(json.loads(f.read_text())['rows']) for f in (path/'conversations').glob('*.json'))
            sentiment_done=0
            for target in sorted((path/'conversations').glob('*.json')):
                record = json.loads(target.read_text())
                for row in record['rows']:
                    if row['message'] not in cache: cache[row['message']] = measure_sentiment(row['message'])['sentiment_score']
                    row['sentiment'] = cache[row['message']]
                    sentiment_done+=1
                    status('sentiment',phase_completed=sentiment_done,phase_total=sentiment_total)
                write(target, record)
        if config.get('export_profile') == 'full_behavioral':
            status('explanation_similarity')
            from .message_similarity import MessageSimilarity
            encoder = None
            for target in sorted((path/'conversations').glob('*.json')):
                record = json.loads(target.read_text())
                previous = {}
                for row in record['rows']:
                    history = previous.setdefault(row['agent'], [])
                    text = row['argument'] if row.get('choice_status') not in ('invalid_json','invalid_schema') else ''
                    if 'explanation_similarity' not in row:
                        value = adjacent = None
                        if text and history:
                            if encoder is None: encoder = MessageSimilarity()
                            scores = encoder.compare([text]*len(history),history)
                            value=max(scores);adjacent=scores[-1]
                        row['explanation_similarity'] = value
                        row['adjacent_explanation_similarity'] = adjacent
                    if text: history.append(text)
                write(target, record)
            del encoder
            gc.collect()
        if 'prediction' in config.get('research', []):
            from .prediction import run as predict, predict_groups
            status('prediction')
            predict(path)
            predict_groups(path)
        if config['mode'] == 'scenario_exploration':
            status('argument_similarity')
            from .exploration_analysis import analyze
            analyze(path)
        if config.get('dataset') and config.get('message_similarity', True):
            status('message_similarity')
            from .message_analysis import analyze as message_analysis
            message_analysis(path,lambda completed,total:status('message_similarity',phase_completed=completed,phase_total=total))
        if config.get('dataset'):
            status('behavioral_analysis')
            from .behavioral import analyze
            analyze(path)
        status('figures')
        from .figures import render
        if config.get('dataset') and config.get('question_figure_profile')=='question_png':
            pass  # The compact report renders the question PNGs once below.
        elif config['mode'] == 'scenario_exploration':
            from .exploration_analysis import plots
            plots(path)
        else:
            render(path)
        if config.get('dataset'):
            status('reports')
            from .behavioral_reports import generate, compare_experiments
            generate(path,reanalyze=False)
            if path.parent.name=='experiments':
                compare_experiments(path.parent,include_current=path) if config.get('question_figure_profile')!='question_png' else None
        status('complete')
    except BaseException as error:
        status('cancelled' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        (path/'error.log').write_text(traceback.format_exc())
        raise
    return path
