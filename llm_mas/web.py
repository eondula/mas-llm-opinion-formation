"""Local dashboard using the same CLI/configuration as terminal runs."""
import json
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse, parse_qs, unquote
import mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from .config import ROOT, resolve, run_path, scenario, validate_scenario, resolve_run_id
from .storage import write
from . import jobs
from .planning import configurations
from .locations import input_path, run_directories, historical_root


def belongs_to_page(name, scope):
    if not name: return False
    if scope not in ('main', 'explore'): return True
    try:
        path=run_path(name)/'config.json'
        if not path.exists():
            path=input_path(Path('results/studies/configurations')/f'{name}.json', ROOT)
        config=json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    exploration=config.get('mode')=='scenario_exploration'
    return exploration if scope=='explore' else not exploration


def display_event(event):
    """Send only fields used by dashboard playback; raw results stay on disk."""
    row=event['row']
    small={k:row[k] for k in ('round','agent','model','strategy','answer','correct','question_id','forced','message','argument','sentiment','deliveries','choice_status','response_id','forced_assignment','forced_compliance','argument_id') if k in row}
    from .choice_output import display_message
    small['display_message'] = display_message(row)
    small['post']={k:row.get('post',{}).get(k) for k in ('stance','margin') if k in row.get('post',{})}
    if row.get('decision_time'):
        small['decision_time']={k:row['decision_time'][k] for k in ('stance','margin','selected_answer','score_preferred_answer')}
    small['stance_source']='decision_time' if row.get('decision_time') else 'post_response_probe'
    if 'diagnostic' in row:
        small['diagnostic']={k:row['diagnostic'].get(k) for k in ('status','stage','margin','stance','strategy','tie','error')}
    return dict(task=event['task'],row=small)


def live_status(status, modified, now=None):
    """Keep elapsed time moving between worker status writes."""
    age=max(0, (time.time() if now is None else now)-modified)
    return dict(status, elapsed_seconds=status.get('elapsed_seconds',0)+age,
                phase_elapsed_seconds=status.get('phase_elapsed_seconds',0)+age,
                seconds_since_update=age)


def active_runs():
    """Discover running CLI studies, including ones started before the server."""
    managed=set(jobs.running(ROOT))
    try:
        output=subprocess.check_output(['ps','-ax','-o','command='],text=True)
    except (OSError,subprocess.SubprocessError):
        return sorted(managed)
    names=set(managed)
    for line in output.splitlines():
        try:
            args=shlex.split(line)
            i=args.index('-m')
            if args[i+1] != 'llm_mas' or args[i+2] not in ('run', 'study'):continue
            name=args[args.index('--run-id')+1]
            path=run_path(name)
            if not (path/'config.json').exists():continue
            status=json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {}
            if status.get('phase') not in ('complete','failed','cancelled'):names.add(name)
        except (ValueError,IndexError,OSError,json.JSONDecodeError):
            continue
    return sorted(names)


def serve(port=8766):
    state={'process':None,'run_id':None}
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, content='application/json', status=200):
            body=body if isinstance(body,bytes) else (json.dumps(body).encode() if content=='application/json' else body.encode())
            self.send_response(status);self.send_header('Cache-Control','no-store');self.send_header('Content-Type',content);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            try:
                parsed=urlparse(self.path)
                if parsed.path=='/api/pilot':
                    from .pilot import preset,assert_lock,available_device
                    from .prompts import templates
                    device=parse_qs(parsed.query).get('device',['auto'])[0]
                    if device=='auto':device=available_device()
                    raw=preset(device=device);config=resolve(raw)
                    try:
                        frozen=assert_lock(config,templates());ready=True;message='Frozen protocol ready'
                    except ValueError as error:
                        frozen={};ready=False;message=str(error)
                    return self.send(dict(config=raw,ready=ready,message=message,protocol_hash=frozen.get('protocol_hash'),conversations=240,paired_blocks=120,logical_rows=2880,generations=2640,generation_stages_per_response=2))
                if parsed.path=='/api/experiments':
                    entries=[]
                    for folder in sorted((ROOT/'results/experiments').glob('*'),key=lambda p:p.stat().st_mtime,reverse=True):
                        if not (folder/'config.json').is_file():continue
                        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').is_file() else {}
                        entries.append(dict(id=folder.name,phase=status.get('phase','starting')))
                    return self.send(entries)
                if parsed.path=='/api/experiment':
                    query=parse_qs(parsed.query)
                    discovered=active_runs()
                    current=jobs.latest(ROOT,discovered) or state['run_id']
                    if current and not (ROOT/'results/experiments'/current).exists():
                        current=None
                    name=query.get('run_id',[current])[0]
                    if name:
                        resolve_run_id(name)
                    folder=ROOT/'results/experiments'/name if name else None
                    config_file=ROOT/'results/experiments/configurations'/f'{name}.json' if name else None
                    proc=state['process'] if name==state['run_id'] else None
                    running=(proc is not None and proc.poll() is None) or name in discovered
                    worker_busy=(state['process'] is not None and state['process'].poll() is None) or bool(discovered)
                    if folder and not folder.exists() and not (config_file and config_file.exists()) and proc is None:
                        return self.send({'error':'Experiment not found'},status=404)
                    status_file=folder/'status.json' if folder else None
                    status=json.loads(status_file.read_text()) if status_file and status_file.is_file() else {'phase':'starting' if running else 'idle'}
                    if running and status_file and status_file.is_file():
                        status=live_status(status,status_file.stat().st_mtime)
                    worker_host=jobs.worker_host(folder) if folder else None
                    remote=bool(worker_host and worker_host!=socket.gethostname())
                    if not running and not remote and folder and status.get('phase') not in ('idle','complete','failed','cancelled'):
                        status=dict(status,phase='failed',error='Experiment worker stopped before completion. See worker.log and error.log in the experiment directory.')
                    summary_file=folder/'summary.json' if folder else None
                    summary=json.loads(summary_file.read_text()).get('rows',[]) if status['phase']=='complete' and summary_file and summary_file.exists() else []
                    return self.send(dict(run_id=name,status=status,running=bool(running),worker_busy=worker_busy,
                                          worker_host=worker_host,remote_progress=remote,can_stop=(proc is not None and proc.poll() is None) or bool(folder and running and jobs.worker(folder) is not None),summary=summary,
                                          output_directory=str(folder.relative_to(ROOT)) if folder else None,
                                          report_available=bool(folder and (folder/'index.html').exists())))
                if parsed.path=='/api/analyses':
                    from .saved_analysis import list_analyses
                    proc=state['process']
                    active=state['run_id'] if proc and proc.poll() is None else None
                    return self.send({'analyses':list_analyses(),'active_run':active})
                if parsed.path=='/api/analysis':
                    from .saved_analysis import catalog
                    name=parse_qs(parsed.query)['run_id'][0]
                    result=catalog(name)
                    proc=state['process'] if state['run_id']==name else None
                    if proc and proc.poll() is not None and result['status'].get('phase') not in ('complete','failed','cancelled'):
                        result['status']={'phase':'failed','error':f'Analysis process exited with code {proc.returncode}. See results/studies/configurations/{name}.log'}
                    return self.send(result)
                if parsed.path=='/api/active-runs':
                    return self.send([name for name in active_runs() if belongs_to_page(name,parse_qs(parsed.query).get('scope',[None])[0])])
                if parsed.path=='/api/runs':
                    query = parse_qs(parsed.query)
                    archive_only = query.get('archive', [None])[0] == 'only'
                    return self.send([p.name for p in run_directories(ROOT)
                                      if (not archive_only or p.is_relative_to(historical_root(ROOT)))
                                      and (p/'config.json').exists()
                                      and belongs_to_page(p.name,query.get('scope',[None])[0])])
                if parsed.path.startswith('/results/'):
                    relative=unquote(parsed.path[len('/results/'):])
                    parts=Path(relative).parts
                    if len(parts)<2 or Path(relative).is_absolute():return self.send({'error':'Not found'},status=404)
                    root=run_path(parts[0]).resolve();target=(root.joinpath(*parts[1:])).resolve()
                    if root not in target.parents or not target.is_file():return self.send({'error':'Not found'},status=404)
                    return self.send(target.read_bytes(),mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
                if parsed.path=='/api/questions':
                    from .datasets import describe
                    query=parse_qs(parsed.query)
                    return self.send(describe(ROOT,query.get('partition',['train'])[0],
                                              query.get('split',['train'])[0],int(query.get('split_seed',['42'])[0])))
                if parsed.path=='/api/scenarios':
                    entries=[]
                    if parse_qs(parsed.query).get('archived') != ['1']:
                        return self.send(entries)
                    for folder in ['controlled','custom']:
                        for p in (historical_root(ROOT)/'scenarios'/folder).glob('*.json'):
                            if p.name!='index.json':entries.append({'path':str(p.relative_to(ROOT)),'data':json.loads(p.read_text())})
                    return self.send(entries)
                if parsed.path=='/api/status':
                    scope=parse_qs(parsed.query).get('scope',[None])[0]
                    discovered=[name for name in active_runs() if belongs_to_page(name,scope)]
                    current=state['run_id'] if belongs_to_page(state['run_id'],scope) else next(iter(discovered),None)
                    name=parse_qs(parsed.query).get('run_id',[current])[0]
                    if name and not belongs_to_page(name,scope):
                        return self.send({'error':'Run belongs to a different dashboard page'},status=400)
                    p=run_path(name) if name else None
                    status=json.loads((p/'status.json').read_text()) if p and (p/'status.json').exists() else {'phase':'idle'}
                    events=[]
                    closure_events=[]
                    if p and (p/'events.jsonl').exists():
                        for line in (p/'events.jsonl').read_text().splitlines():
                            try:
                                event = json.loads(line)
                                if "row" in event: events.append(display_event(event))
                                if event.get('kind') == 'termination_check' and event.get('policy') == 'agent_closure':
                                    closure_events.append({k:event.get(k) for k in ('task','round','reason','proposal','unavailable')} | {
                                        phase:[{k:v.get(k) for k in ('agent','action','reason','valid')} for v in event.get(phase,[])]
                                        for phase in ('proposals','confirmations')})
                            except json.JSONDecodeError:pass
                    proc=state['process'] if name==state['run_id'] else None
                    if p and (p/'conversations').exists():
                        completed={f.stem:json.loads(f.read_text()) for f in (p/'conversations').glob('*.json')}
                        events=[e for e in events if e['task'] not in completed]
                        for task, record in completed.items():
                            events.extend(display_event(dict(task=task,row=row)) for row in record['rows'])
                        events.sort(key=lambda e:(e['task'],e['row']['round']))
                    if proc is not None:
                        code=proc.poll()
                        if code is None and status['phase']=='idle':status={'phase':'starting'}
                        elif code is not None and status['phase'] not in ('complete','failed','cancelled'):
                            log_path=input_path(Path('results/studies/configurations')/f'{name}.log', ROOT)
                            detail=log_path.read_text()[-4000:] if log_path.exists() else ''
                            status=dict(status,phase='failed',error=f'Run process exited with code {code}. '+detail)
                    running = (proc is not None and proc.poll() is None) or name in discovered
                    if running and p and (p/'status.json').exists():
                        status = live_status(status, (p/'status.json').stat().st_mtime)
                        log_path=input_path(Path('results/studies/configurations')/f'{name}.log', ROOT)
                        if log_path.exists():
                            with log_path.open('rb') as log:
                                log.seek(max(0, log_path.stat().st_size-2000))
                                status['log_tail']=log.read().decode('utf-8',errors='replace')
                    config=json.loads((p/'config.json').read_text()) if p and (p/'config.json').exists() else None
                    tasks=json.loads((p/'tasks.json').read_text()) if p and (p/'tasks.json').exists() else []
                    evaluation=json.loads((p/'analysis/evaluation.json').read_text()).get('summary') if p and (p/'analysis/evaluation.json').is_file() else None
                    return self.send(dict(report_available=bool(p and (p/'index.html').exists()),evaluation=evaluation,worker_busy=state['process'] is not None and state['process'].poll() is None,run_id=name,status=status,events=events,closure_events=closure_events,config=config,tasks=tasks,running=(proc is not None and proc.poll() is None) or name in discovered,can_stop=proc is not None and proc.poll() is None))
                assets={'/':('dashboard/study.html','text/html'),'/study.html':('dashboard/study.html','text/html'),'/conversation.html':('dashboard/study.html','text/html'),'/study.js':('dashboard/study.js','text/javascript'),'/study.css':('dashboard/study.css','text/css'),'/styles.css':('dashboard/styles.css','text/css'),'/site.css':('dashboard/site.css','text/css'),'/site.js':('dashboard/site.js','text/javascript'),'/model-controls.js':('dashboard/model-controls.js','text/javascript'),'/experiment.html':('dashboard/experiment.html','text/html'),'/experiment.js':('dashboard/experiment.js','text/javascript'),'/experiment.css':('dashboard/experiment.css','text/css')}
                if self.path in assets:
                    filename,mime=assets[self.path];return self.send((ROOT/filename).read_bytes(),mime)
                return self.send({'error':'Not found'},status=404)
            except Exception as error:self.send({'error':str(error)},status=400)
        def do_POST(self):
            try:
                # Reject cross-origin browser submissions to this local execution endpoint.
                origin=self.headers.get('Origin')
                if origin and origin not in (f'http://localhost:{port}',f'http://127.0.0.1:{port}'):raise ValueError('Invalid origin')
                length=int(self.headers.get('Content-Length','0'))
                if length>2_000_000:raise ValueError('Request too large')
                data=json.loads(self.rfile.read(length))
                if self.path=='/api/scenarios':
                    raise ValueError('Authored scenarios are archived. Configure a StrategyQA experiment instead.')
                if self.path=='/api/preview':
                    from .prompts import templates,initial
                    config=resolve(data)
                    if config.get('dataset'):
                        from .question_policy import QuestionResponsePolicy
                        from .engine import names_for
                        texts=templates();names=names_for(config['agents'])
                        prompts=[]
                        for task in config['tasks']:
                            policy=QuestionResponsePolicy(config,task,texts,names,config['models'])
                            prompts.extend(dict(question_id=task['id'],agent=names[i],history=policy.initial(i),
                                                response=policy.response_prompt(0,i,[]),
                                                update_response=policy.response_prompt(1,i,config['agents']>1 and config['schedule']!='no_peer'))
                                           for i in range(config['agents']))
                        return self.send(dict(config=config,prompts=prompts,measurement_probe=texts['qa_probe'].format(candidate_1='Yes',candidate_2='No')))
                    if config['mode']=='scenario_exploration':
                        from .exploration import initial as explore_initial, response_prompt
                        texts=templates(); scenario=config['scenarios'][0]
                        return self.send(dict(config=config,prompts=[dict(history=explore_initial(config,scenario,texts,i),response=response_prompt(config,texts,0,i,False)) for i in range(config['agents'])],diagnostic=texts['exploration_probe'].format(candidate_1=scenario['strategy_1']['candidate'],candidate_2=scenario['strategy_2']['candidate']) if config['stance_diagnostic'] else None))
                    return self.send(dict(config=config,prompts=[initial(s,templates(),config['presentation_order'],config['agents'],persona=i%2+1 if config.get('persona') else None, prompt_style=config.get('prompt_style','original'), joint_task_brief=config.get('joint_task_brief') if config['mode']=='joint_task' else None, agent_name=(f'Agent {chr(65+i)}' if i<26 else f'Agent {i+1}') if config.get('prompt_style','original')!='original' or config['mode']=='joint_task' else None) for s in config['scenarios'] for i in range(config['agents'] if config.get('persona') or config.get('prompt_style','original')!='original' or config['mode']=='joint_task' else 1)]))
                with lock:
                    proc=state['process']
                    if self.path=='/api/stop':
                        name=data.get('run_id') or state['run_id'] or jobs.latest(ROOT,active_runs())
                        if name:resolve_run_id(name)
                        if name==state['run_id'] and proc and proc.poll() is None:
                            proc.send_signal(signal.SIGINT)
                        elif not name or not jobs.stop(ROOT/'results/experiments'/name):
                            raise ValueError('No verified active worker is available for this run')
                        return self.send({'stopping':True})
                    if self.path=='/api/experiment':
                        if proc and proc.poll() is None or active_runs():
                            raise ValueError('An experiment is already active')
                        config=resolve(dict(data,mode='experiment',output_layout='experiment'))
                        if not config.get('dataset'):
                            raise ValueError('Select StrategyQA for the current experiment workflow')
                        if config.get('pilot_protocol'):
                            from .pilot import assert_lock
                            from .prompts import templates
                            assert_lock(config,templates())
                        if not config.get('pilot_protocol') and (config.get('grid') or config.get('agent_counts') or config.get('model_configurations')):
                            raise ValueError('Use explicit experiment settings and optional schedule controls')
                        name=resolve_run_id(config.pop('run_id',None))
                        folder=ROOT/'results/experiments'/name
                        if folder.exists() or run_path(name).exists():
                            raise ValueError('Experiment ID already exists; choose another ID')
                        configuration=ROOT/'results/experiments/configurations'/f'{name}.json'
                        write(configuration,config)
                        # A log lives with the experiment even while the model loads.
                        from .launch import reserve
                        launch_token=reserve(folder,config)
                        write(folder/'status.json',dict(phase='starting',completed=0,total=sum(len(c['scenarios'])*(c['rounds']+1)*c['agents'] for c in configurations(config))))
                        with (folder/'worker.log').open('w') as log:
                            command=[sys.executable,'-u','-m','llm_mas','run','--config',str(configuration),'--run-id',name,'--launch-token',launch_token]
                            try:
                                process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                                jobs.register(folder,process,command)
                            except BaseException as error:
                                write(folder/'status.json',dict(phase='failed',error=str(error)))
                                raise
                        state.update(process=process,run_id=name)
                        return self.send(dict(run_id=name,output_directory=str(folder.relative_to(ROOT))))
                    if self.path=='/api/analyze':
                        if proc and proc.poll() is None:raise ValueError('A run or analysis is already active')
                        from .saved_analysis import validate
                        validate(data)
                        name=resolve_run_id(data.pop('run_id',None))
                        if run_path(name).exists():raise ValueError('Output ID already exists')
                        path=ROOT/'results/studies/configurations'/f'{name}.json'
                        write(path,data)
                        with path.with_suffix('.log').open('w') as log:
                            process=subprocess.Popen([sys.executable,'-u','-m','llm_mas','analyze','--config',str(path),'--run-id',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                        state.update(process=process,run_id=name)
                        return self.send({'run_id':name})
                    if self.path=='/api/run':
                        if proc and proc.poll() is None:raise ValueError('A run is already active')
                        config=resolve(data)
                        name=resolve_run_id(config.pop('run_id', None))
                        if run_path(name).exists(): raise ValueError('Run ID already exists; use another ID or resume the saved run')
                        path=ROOT/'results/studies/configurations'/f'{name}.json'
                        write(path,config)
                        log=path.with_suffix('.log').open('w')
                        command=[sys.executable,'-u','-m','llm_mas','run','--config',str(path),'--run-id',name]
                        state.update(process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),run_id=name)
                        log.close()
                        return self.send({'run_id':name})
                self.send({'error':'Not found'},status=404)
            except Exception as error:self.send({'error':str(error)},status=400)
    print(f'Opinion Lab: http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
