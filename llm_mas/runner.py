import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from .config import resolve, digest, run_path, ROOT, resolve_run_id
from .prompts import templates
from .engine import conversation


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def execute(raw, name=None, resume=False):
    config = resolve(raw)
    name = resolve_run_id(name or config.pop('run_id', None))
    path = run_path(name)
    texts = templates()
    identity = digest(dict(config=config, prompts=texts))
    if path.exists():
        if not resume: raise ValueError(f'Run already exists: {path}')
        manifest = json.loads((path/'manifest.json').read_text())
        if manifest.get('source_hashes') != {str(p.relative_to(ROOT)):digest(p.read_text()) for p in (ROOT/'llm_mas').glob('*.py')}: raise ValueError('Source code changed since run; start a new run instead of resuming')
        if manifest['input_hash'] != identity: raise ValueError('Resume inputs differ from saved configuration/prompts')
    else:
        path.mkdir(parents=True)
        write(path/'config.json', config)
        write(path/'inputs/prompts.json', texts)
        write(path/'inputs/scenarios.json', config['scenarios'])
        write(path/'manifest.json', dict(input_hash=identity, schema_version=1, created=datetime.now().isoformat(),
                                       resume_unit='completed conversation', source_hashes={str(p.relative_to(ROOT)):digest(p.read_text()) for p in (ROOT/'llm_mas').glob('*.py')}))
    start = time.monotonic()
    from .planning import configurations
    tasks = [resolve(c) for c in configurations(config)]
    if not tasks: raise ValueError('No compatible model/agent configurations')
    write(path/'tasks.json', tasks)
    total = sum(len(c['scenarios'])*(c['rounds']+1)*c['agents'] for c in tasks)
    done = 0
    runtimes = {}
    from .progress import Progress
    tracker=Progress()
    phase_counts={}
    def status(phase, **extra):
        metrics=tracker.update(phase,extra.pop('phase_completed',done if phase=='conversation' else None),extra.pop('phase_total',total if phase=='conversation' else None))
        write(path/'status.json', dict(metrics, completed=done,total=total,**extra))
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
        for ti, active in enumerate(tasks):
            for si, scenario in enumerate(active['scenarios']):
                target = path/'conversations'/f'{ti:03d}_{si:03d}.json'
                if target.exists():
                    done += (active['rounds']+1)*active['agents']
                    continue
                for name_ in set(active['models']):
                    if name_ not in runtimes:
                        status('loading', model=name_)
                        runtimes[name_] = Runtime(name_, active['device'])
                base = done
                def emit(**event):
                    nonlocal done
                    done = base+event['completed']
                    with (path/'events.jsonl').open('a') as out:
                        out.write(json.dumps(dict(task=target.stem, **event))+'\n')
                    status('conversation', task=target.stem)
                    elapsed = time.monotonic()-start
                    print(f'[{"="*int(30*done/total):30}] {done}/{total} elapsed {elapsed:.0f}s ETA {elapsed*(total-done)/done:.0f}s', flush=True)
                rows = conversation(active, scenario, texts, runtimes, emit)
                write(target, dict(config=active, scenario=scenario, rows=rows))
        if config.get('research'):
            from .research import run as run_research
            for model_name in {m for task in tasks for m in task['models']}:
                if model_name not in runtimes: runtimes[model_name] = Runtime(model_name, config['device'])
            status('research')
            run_research(path, config, texts, runtimes, research_progress)
        runtimes.clear()
        import gc
        gc.collect()
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
        if 'prediction' in config.get('research', []):
            from .prediction import run as predict, predict_groups
            status('prediction')
            predict(path)
            predict_groups(path)
        status('figures')
        from .figures import render
        render(path)
        status('complete')
    except BaseException as error:
        status('cancelled' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        (path/'error.log').write_text(traceback.format_exc())
        raise
    return path
