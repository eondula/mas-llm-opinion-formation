"""Independent, traceable analysis jobs over immutable saved conversations."""
from .trajectories import read_conversation
import csv
import json
import shutil
from pathlib import Path
from .config import ROOT, run_path, resolve_run_id, digest
from .storage import write
from .locations import run_directories

MODULES = {'replay', 'mechanistic', 'heads', 'source_spans', 'continuation'}


def read_record(run_id, task):
    root = run_path(run_id)
    if not task or Path(task).name != task or task in ('.', '..'):
        raise ValueError('Select a saved conversation')
    file = root/'conversations'/f'{task}.json'
    if not file.is_file():
        raise ValueError('Saved conversation not found')
    return root, file, read_conversation(file)


def list_analyses():
    items=[]
    for directory in run_directories(ROOT):
        file=directory/'manifest.json'
        try:
            manifest=json.loads(file.read_text())
            if manifest.get('kind')!='saved_run_analysis':continue
            status_file=file.parent/'status.json'
            status=json.loads(status_file.read_text()) if status_file.exists() else {}
            items.append(dict(run_id=file.parent.name,source_run=manifest.get('source_run'),
                              phase=status.get('phase','starting')))
        except (OSError,json.JSONDecodeError):continue
    return items


def catalog(run_id):
    root = run_path(run_id)
    records = []
    for file in sorted((root/'conversations').glob('*.json')):
        r = read_conversation(file)
        records.append(dict(task=file.stem, title=r['scenario']['title'], config=r['config'],
            events=[dict(agent=x['agent'], round=x['round'], model=x['model']) for x in r['rows'] if x['round'] > 0 and x.get('deliveries')]))
    def url(p):
        return '/results/'+run_id+'/'+p.relative_to(root).as_posix()
    figures = [dict(name=str(p.relative_to(root/'figures')), png=url(p),
                   pdf=url(p.with_suffix('.pdf')) if p.with_suffix('.pdf').exists() else None,
                   csv=url(p.with_suffix('.csv')) if p.with_suffix('.csv').exists() else None)
               for p in sorted((root/'figures').rglob('*.png'))
               if not (p.parent==root/'figures' and (root/'figures/appendix'/p.name).exists())]
    tables = []
    for file in sorted((root/'tables').glob('*.csv')):
        with file.open(newline='') as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            rows = []
            for i, row in enumerate(reader):
                if i >= 20: break
                rows.append(row)
        tables.append(dict(name=file.stem, csv=url(file), header=header, rows=rows))
    status = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    coverage=json.loads((root/'figures/coverage.json').read_text()) if (root/'figures/coverage.json').exists() else {}
    evaluation=json.loads((root/'analysis/evaluation.json').read_text()).get('summary') if (root/'analysis/evaluation.json').is_file() else None
    return dict(coverage=coverage, run_id=run_id, records=records, figures=figures, tables=tables, status=status,
                dataset=records[0]['config'].get('dataset') if records else None, evaluation=evaluation,
                zip=url(root/'paper_outputs.zip') if (root/'paper_outputs.zip').exists() else None)


def validate(options):
    root, file, record = read_record(options['source_run'], options['task'])
    modules = options.get('modules', [])
    if not isinstance(modules, list) or set(modules)-MODULES:
        raise ValueError('Unsupported saved-run analysis')
    if modules and record.get('dataset'):
        raise ValueError('Historical intervention analyses are not StrategyQA update-model analyses. Inspect the saved trajectories and correctness tables instead.')
    if modules:
        if not any(r['agent']==options.get('agent') and r['round']==options.get('round') and r.get('deliveries') for r in record['rows']):
            raise ValueError('Select an agent/round with peer delivery')
        if not (root/'inputs/prompts.json').exists():
            raise ValueError('Original saved prompts are required for model analyses')
    layers = options.get('layers')
    if layers is not None and (not isinstance(layers,list) or not layers or any(type(x)!=int or x<0 for x in layers)):
        raise ValueError('Layers must be nonnegative, zero-based integers')
    if options.get('device','auto') not in ('auto','cpu','cuda','mps'):
        raise ValueError('Invalid device')
    if options.get('compare_run'):
        _, _, other = read_record(options['compare_run'], options.get('compare_task'))
        if record['scenario'] != other['scenario']:
            raise ValueError('Compare conversations with the same scenario definition and strategy meanings')
    return root, file, record


def execute(options, name):
    root, source, record = validate(options)
    path = run_path(resolve_run_id(name))
    path.mkdir(parents=True, exist_ok=False)
    import time
    from .progress import Progress, JobProgress
    tracker=Progress()
    job_progress=JobProgress((["loading"] if options.get("modules") else [])+options.get("modules",[])+["figures"]+(["comparison"] if options.get("compare_run") else [])+["packaging"])
    def status(phase, **extra):
        metrics=tracker.update(phase,extra.pop('phase_completed',None),extra.pop('phase_total',None))
        overall=job_progress.update(phase,metrics['phase_completed'],metrics['phase_total'])
        write(path/'status.json',dict(metrics,**overall,updated_at=time.time(),**extra))
    config = dict(record['config'], research=options.get('modules', []),
                  research_selection={'agent':options.get('agent'), 'round':options.get('round')},
                  research_events_per_conversation=1, patch_layers=options.get('layers'),
                  continuation_all_layers=True)
    write(path/'config.json', config)
    write(path/'conversations'/source.name, record)
    provenance = dict(kind='saved_run_analysis', source_run=options['source_run'], task=options['task'],
                      source_sha256=digest(source.read_text()), options=options,
                      source_hashes={str(p.relative_to(ROOT)):digest(p.read_text()) for p in (ROOT/'llm_mas').glob('*.py')})
    if (root/'manifest.json').exists():provenance['source_manifest']=json.loads((root/'manifest.json').read_text())
    write(path/'manifest.json', provenance)
    if (root/'inputs/prompts.json').exists():
        write(path/'inputs/prompts.json', json.loads((root/'inputs/prompts.json').read_text()))
    try:
        if config['research']:
            status('loading')
            from .runtime import Runtime
            models = set(record['config']['models']) if 'continuation' in config['research'] else {
                r['model'] for r in record['rows'] if r['agent']==options['agent']}
            runtimes = {m:Runtime(m, options.get('device','auto')) for m in models}
            for runtime in runtimes.values():
                if set(config['research'])-{'replay'} and any(x>=len(runtime.model.model.layers) for x in (config['patch_layers'] or [])):
                    raise ValueError('Selected layer is outside the model')
            from .research import run
            status(config['research'][0], detail='Evaluating selected saved history')
            run(path,config,json.loads((root/'inputs/prompts.json').read_text()),runtimes,
                lambda module,model,scenario:status(module,detail='Processing selected history') if module=='replay' else None,
                layer_progress=lambda module,done,total,layer:status(module,phase_completed=done,phase_total=total,
                    detail=f'{done}/{total} layers completed'+(f' · Last layer: {layer}' if layer is not None else '')))
            runtimes.clear()
        # Reuse existing measurements for this conversation unless explicitly recomputed.
        for source_measure in (root/'measurements').glob('*.json'):
            if source_measure.stem in config['research']:continue
            items=json.loads(source_measure.read_text())
            selected=[r for r in items if r.get('task')==options['task'] or (r.get('scenario')==record['scenario']['id'] and r.get('model') in record['config']['models'])]
            if selected:write(path/'measurements'/source_measure.name,selected)
        status('figures')
        from .figures import render
        render(path)
        if options.get('compare_run'):
            status('comparison',detail='Preparing comparison figures and CSV data')
            _, second_file, other = read_record(options['compare_run'],options['compare_task'])
            write(path/'comparison_source.json',dict(run_id=options['compare_run'], task=options['compare_task'],
                  source_sha256=digest(second_file.read_text()),record=other))
            compare(path,record,other,options['source_run'],options['compare_run'])
        # Include all added comparison exports in the study bundle.
        status('packaging',detail='Finalizing downloadable outputs')
        import zipfile
        with zipfile.ZipFile(path/'paper_outputs.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for directory in ('figures','tables'):
                for f in (path/directory).rglob('*'):
                    if f.is_file():archive.write(f,f.relative_to(path))
        status('complete')
    except BaseException as error:
        status('cancelled' if isinstance(error,KeyboardInterrupt) else 'failed', error=str(error))
        raise
    return path


def compare(path,a,b,name_a,name_b):
    import matplotlib.pyplot as plt
    from .tables import table
    rows=[]
    for side,name,record in [('A',name_a,a),('B',name_b,b)]:
        for r in record['rows']:
            rows.append(dict(side=side,run=name,agent=r['agent'],model=r['model'],round=r['round'],
                strategy=r['strategy'],decision_margin=(r.get('decision') or {}).get('margin'),
                post_margin=r['post']['margin'],post_stance=r['post']['stance'],sentiment=r.get('sentiment')))
    table(path/'tables/comparison_trajectories',rows)
    differences=[]
    lookup={(r['agent'],r['round']):r for r in rows if r['side']=='B'}
    for r in rows:
        other=lookup.get((r['agent'],r['round']))
        if r['side']=='A' and other:
            differences.append(dict(agent=r['agent'],round=r['round'],model_A=r['model'],model_B=other['model'],
                post_margin_B_minus_A=other['post_margin']-r['post_margin'],strategy_agrees=r['strategy']==other['strategy']))
    table(path/'tables/comparison_matched_rounds',differences)
    table(path/'tables/comparison_conditions',[dict(side=side,run=name,**{k:json.dumps(v) for k,v in rec['config'].items() if k in ('models','agents','schedule','initialization','argument_mode','presentation_order','rounds')}) for side,name,rec in [('A',name_a,a),('B',name_b,b)]])
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for side,name in [('A',name_a),('B',name_b)]:
        for agent in dict.fromkeys(r['agent'] for r in rows if r['side']==side):
            subset=[r for r in rows if r['side']==side and r['agent']==agent]
            for ax,field in zip(axes,('decision_margin','post_margin')):
                pts=[r for r in subset if r[field] is not None]
                ax.plot([r['round'] for r in pts],[r[field] for r in pts],linestyle='-' if side=='A' else '--',label=f'{side}: {name} / {agent}')
    for ax,title in zip(axes,('Decision margin','Post-response margin')):
        ax.set(title=title,xlabel='Round',ylabel='Raw likelihood margin');ax.legend(fontsize=7)
    for ext in ('png','pdf'):fig.savefig(path/'figures'/('comparison.'+ext),dpi=160)
    plt.close(fig)
    shutil.copyfile(path/'tables/comparison_trajectories.csv',path/'figures/comparison.csv')
