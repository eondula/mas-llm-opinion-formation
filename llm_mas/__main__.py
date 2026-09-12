import argparse
import json
from pathlib import Path
from .config import ROOT, DEFAULT_MODEL, resolve, scenario, run_path, validate_scenario

def main():
    p=argparse.ArgumentParser(description='Opinion formation study runner')
    sub=p.add_subparsers(dest='command',required=True)
    preview=sub.add_parser('preview')
    preview.add_argument('--model',default=DEFAULT_MODEL)
    preview.add_argument('--agents',type=int,default=2)
    preview.add_argument('--models',nargs='+',help='One checkpoint per agent; overrides --model')
    preview.add_argument('--agent-configuration',choices=['homogeneous','heterogeneous'])
    preview.add_argument('--scenario',default='storage')
    preview.add_argument('--rounds',type=int,default=20)
    preview.add_argument('--argument-mode',choices=['predefined','agent_selected','generated','conversational','combined_fixed','combined_reasoning'],default='predefined')
    preview.add_argument('--initialization',default='natural',help='natural or comma-separated 1,2')
    preview.add_argument('--schedule',default='synchronous',choices=['synchronous','sequential','reverse_sequential','no_peer'])
    preview.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
    preview.add_argument('--no-sentiment',action='store_true')
    for parser in (preview,sub.add_parser('study')):
        if parser != preview: parser.add_argument('--config',required=True,help='JSON configuration')
        parser.add_argument('--run-id')
        parser.add_argument('--dry-run',action='store_true')
    analyze=sub.add_parser('analyze');analyze.add_argument('--config',required=True);analyze.add_argument('--run-id',required=True)
    resume=sub.add_parser('resume');resume.add_argument('--run-id',required=True)
    figures=sub.add_parser('figures');figures.add_argument('--run-id',required=True)
    dashboard=sub.add_parser('dashboard');dashboard.add_argument('--port',type=int,default=8766)
    scenarios=sub.add_parser('scenarios');scenarios.add_argument('action',choices=['list','create','validate'])
    scenarios.add_argument('--id',help='ID for a new scenario');scenarios.add_argument('--template',default='storage');scenarios.add_argument('--output');scenarios.add_argument('--file')
    args=p.parse_args()
    if args.command=='analyze':
        from .saved_analysis import execute
        return execute(json.loads(Path(args.config).read_text()),args.run_id)
    if args.command=='dashboard':
        from .web import serve
        return serve(args.port)
    if args.command=='scenarios':
        if args.action=='list':
            for f in sorted((ROOT/'scenarios').glob('*/*.json')):
                if f.name!='index.json': print(f.relative_to(ROOT))
        elif args.action=='create':
            if not args.output: p.error('--output required')
            template=scenario(args.template)
            data={k:template[k] for k in ('title','strategy_1','strategy_2')}
            data['id']=args.id or Path(args.output).stem
            validate_scenario(data)
            target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('x') as out: json.dump(data,out,indent=2)
        else:
            if not args.file:p.error('--file required')
            validate_scenario(json.loads(Path(args.file).read_text()));print('Valid scenario')
        return
    if args.command=='figures':
        from .figures import render
        return render(run_path(args.run_id))
    if args.command=='resume': raw=json.loads((run_path(args.run_id)/'config.json').read_text())
    elif args.command=='study':raw=json.loads(Path(args.config).read_text());raw.setdefault('mode','study')
    else:
        raw=dict(models=args.models or [args.model],agents=args.agents,agent_configuration=args.agent_configuration,scenarios=[args.scenario],rounds=args.rounds,argument_mode=args.argument_mode,
                 initialization='natural' if args.initialization=='natural' else [int(x) for x in args.initialization.split(',')],
                 schedule=args.schedule,device=args.device,sentiment=not args.no_sentiment)
    config=resolve(raw)
    if getattr(args,'dry_run',False):
        from .prompts import templates,initial
        from .planning import configurations
        planned=list(configurations(config))
        print(json.dumps(dict(conversations=sum(len(c['scenarios']) for c in planned),configuration_count=len(planned),config=config,prompt_preview=initial(config['scenarios'][0],templates(),config['presentation_order'],config['agents'])),indent=2));return
    from .runner import execute
    print(execute(config,args.run_id,resume=args.command=='resume'))

if __name__=='__main__':main()
