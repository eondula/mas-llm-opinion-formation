"""Local dashboard using the same CLI/configuration as terminal runs."""
import json
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse, parse_qs, unquote
import mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from .config import ROOT, resolve, run_path, scenario, validate_scenario, resolve_run_id
from .runner import write


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
                if parsed.path=='/api/runs':
                    return self.send([p.name for p in sorted((ROOT/'results/studies').glob('*')) if (p/'config.json').exists()])
                if parsed.path.startswith('/results/'):
                    relative=unquote(parsed.path[len('/results/'):])
                    root=(ROOT/'results/studies').resolve();target=(root/relative).resolve()
                    if root not in target.parents or not target.is_file():return self.send({'error':'Not found'},status=404)
                    return self.send(target.read_bytes(),mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
                if self.path=='/api/scenarios':
                    entries=[]
                    for folder in ['controlled','custom']:
                        for p in (ROOT/'scenarios'/folder).glob('*.json'):
                            if p.name!='index.json':entries.append({'path':str(p.relative_to(ROOT)),'data':json.loads(p.read_text())})
                    return self.send(entries)
                if parsed.path=='/api/status':
                    name=parse_qs(parsed.query).get('run_id',[state['run_id']])[0];p=run_path(name) if name else None
                    status=json.loads((p/'status.json').read_text()) if p and (p/'status.json').exists() else {'phase':'idle'}
                    events=[]
                    if p and (p/'events.jsonl').exists():
                        for line in (p/'events.jsonl').read_text().splitlines():
                            try: events.append(json.loads(line))
                            except json.JSONDecodeError:pass
                    proc=state['process'] if name==state['run_id'] else None
                    if p and (p/'conversations').exists():
                        completed={f.stem:json.loads(f.read_text()) for f in (p/'conversations').glob('*.json')}
                        events=[e for e in events if e['task'] not in completed]
                        for task, record in completed.items():
                            events.extend(dict(task=task,row=row) for row in record['rows'])
                        events.sort(key=lambda e:(e['task'],e['row']['round']))
                    if proc is not None:
                        code=proc.poll()
                        if code is None and status['phase']=='idle':status={'phase':'starting'}
                        elif code is not None and status['phase'] not in ('complete','failed','cancelled'):
                            log_path=ROOT/'results/studies/configurations'/f'{name}.log'
                            detail=log_path.read_text()[-4000:] if log_path.exists() else ''
                            status=dict(status,phase='failed',error=f'Run process exited with code {code}. '+detail)
                    config=json.loads((p/'config.json').read_text()) if p and (p/'config.json').exists() else None
                    tasks=json.loads((p/'tasks.json').read_text()) if p and (p/'tasks.json').exists() else []
                    return self.send(dict(run_id=name,status=status,events=events,config=config,tasks=tasks,running=proc is not None and proc.poll() is None))
                assets={'/':('dashboard/study.html','text/html'),'/study.js':('dashboard/study.js','text/javascript'),'/styles.css':('dashboard/styles.css','text/css'),'/study.css':('dashboard/study.css','text/css'),'/docs.html':('dashboard/docs.html','text/html')}
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
                    validate_scenario(data)
                    run_path(data['id'])
                    target=ROOT/'scenarios/custom'/f"{data['id']}.json"
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with target.open('x') as out:json.dump(data,out,indent=2)
                    return self.send({'saved':str(target.relative_to(ROOT))})
                if self.path=='/api/preview':
                    from .prompts import templates,initial
                    config=resolve(data)
                    return self.send(dict(config=config,prompts=[initial(s,templates(),config['presentation_order'],config['agents']) for s in config['scenarios']]))
                with lock:
                    proc=state['process']
                    if self.path=='/api/stop':
                        if proc and proc.poll() is None:proc.send_signal(signal.SIGINT)
                        return self.send({'stopping':True})
                    if self.path=='/api/run':
                        if proc and proc.poll() is None:raise ValueError('A run is already active')
                        config=resolve(data)
                        name=resolve_run_id(config.pop('run_id', None))
                        if run_path(name).exists(): raise ValueError('Run ID already exists; use another ID or resume the saved run')
                        path=ROOT/'results/studies/configurations'/f'{name}.json'
                        write(path,config)
                        log=path.with_suffix('.log').open('w')
                        command=[sys.executable,'-m','llm_mas','study','--config',str(path),'--run-id',name]
                        state.update(process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),run_id=name)
                        log.close()
                        return self.send({'run_id':name})
                self.send({'error':'Not found'},status=404)
            except Exception as error:self.send({'error':str(error)},status=400)
    print(f'Opinion Lab: http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
