import argparse
import json
from pathlib import Path
from .config import ROOT, DEFAULT_MODEL, resolve, scenario, run_path, validate_scenario, input_path

def main():
    p=argparse.ArgumentParser(description='Opinion formation study runner')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('pilot-freeze',help='Freeze the reviewed pilot protocol')
    dataset=sub.add_parser('dataset', help='Prepare or inspect StrategyQA data')
    dataset.add_argument('action', choices=['download','info'])
    preview=sub.add_parser('preview')
    preview.add_argument('--prompt-style',choices=['original','open_discussion','topic_discussion','discussion_then_decision'],default='original')
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
    for parser in (preview,sub.add_parser('run', help='Run a configuration-driven experiment'),sub.add_parser('study')):
        if parser != preview:
            parser.add_argument('--config',required=True,help='JSON configuration')
            parser.add_argument('--device',choices=['auto','cpu','cuda','mps'],help='Override configuration device')
        parser.add_argument('--run-id')
        parser.add_argument('--dry-run',action='store_true')
        parser.add_argument('--repetitions', type=int, help='Independent realizations per configuration/task')
        parser.add_argument('--seed', type=int, help='Base seed for sampled generation')
    reports=sub.add_parser('reports',help='Regenerate saved StrategyQA reports without model inference')
    report_target=reports.add_mutually_exclusive_group(required=True);report_target.add_argument('--run-id');report_target.add_argument('--all',action='store_true')
    behavioral=sub.add_parser('behavioral',help='Analyze a saved StrategyQA experiment without model inference');behavioral.add_argument('--run-id',required=True)
    analyze=sub.add_parser('analyze');analyze.add_argument('--config',required=True);analyze.add_argument('--run-id',required=True)
    analysis_c=sub.add_parser('analyze-c', help='Analyze Experiment C in its existing output folders')
    analysis_c.add_argument('--run-id',required=True)
    analysis_c.add_argument('--stage',choices=['behavioral','all'],default='behavioral')
    analysis_c.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    analysis_c.add_argument('--tolerance',type=float,default=.01,help='Maximum replay/self-patch margin error')
    resume=sub.add_parser('resume');resume.add_argument('--run-id',required=True)
    figures=sub.add_parser('figures');figures.add_argument('--run-id',required=True)
    dashboard=sub.add_parser('dashboard');dashboard.add_argument('--port',type=int,default=8766)
    scenarios=sub.add_parser('scenarios');scenarios.add_argument('action',choices=['list','create','validate'])
    scenarios.add_argument('--id',help='ID for a new scenario');scenarios.add_argument('--template',default='storage');scenarios.add_argument('--output');scenarios.add_argument('--file')
    args=p.parse_args()
    if args.command=='pilot-freeze':
        from .pilot import freeze
        print(json.dumps(freeze(),indent=2));return
    if args.command=='dataset':
        from .datasets import download, describe
        info=download(ROOT) if args.action=='download' else describe(ROOT)
        print(json.dumps({key:value for key,value in info.items() if key!='questions'},indent=2))
        return
    if args.command=='reports':
        from .behavioral_reports import generate, rebuild_all, compare_experiments
        if args.all:
            rebuild_all(ROOT)
        else:
            folder=run_path(args.run_id);generate(folder)
            if folder.parent.name=='experiments':compare_experiments(folder.parent)
            print(folder/'index.html')
        return
    if args.command=='behavioral':
        folder=run_path(args.run_id)
        from .message_analysis import analyze as message_analysis
        message_analysis(folder)
        from .behavioral_reports import generate, compare_experiments
        generate(folder)
        if folder.parent.name=='experiments':compare_experiments(folder.parent)
        print(folder/'index.html')
        return
    if args.command=='analyze-c':
        from .experiment_c_analysis import execute
        return execute(run_path(args.run_id),args.stage,args.device,args.tolerance)
    if args.command=='analyze':
        from .saved_analysis import execute
        return execute(json.loads(input_path(args.config).read_text()),args.run_id)
    if args.command=='dashboard':
        from .web import serve
        return serve(args.port)
    if args.command=='scenarios':
        if args.action=='list':
            from .locations import historical_root
            for f in sorted((historical_root(ROOT)/'scenarios').glob('*/*.json')):
                if f.name!='index.json': print(f.relative_to(ROOT))
        elif args.action=='create':
            p.error('Authored scenarios are archived. Use dataset download and a StrategyQA experiment configuration.')
        else:
            if not args.file:p.error('--file required')
            validate_scenario(json.loads(Path(args.file).read_text()));print('Valid scenario')
        return
    if args.command=='figures':
        from .figures import render
        folder=run_path(args.run_id)
        render(folder)
        if json.loads((folder/'config.json').read_text()).get('dataset'):
            from .behavioral_reports import generate, compare_experiments
            generate(folder)
            if folder.parent.name=='experiments':compare_experiments(folder.parent)
        return
    if args.command=='resume': raw=json.loads((run_path(args.run_id)/'config.json').read_text())
    elif args.command in ('run', 'study'):
        raw=json.loads(input_path(args.config).read_text())
        if raw.get('pilot_protocol'):
            from .pilot import preset
            raw=dict(preset(),**raw)
        raw.setdefault('mode', 'experiment' if args.command == 'run' else 'study')
    else:
        raw=dict(prompt_style=args.prompt_style,models=args.models or [args.model],agents=args.agents,agent_configuration=args.agent_configuration,scenarios=[args.scenario],rounds=args.rounds,argument_mode=args.argument_mode,
                 initialization='natural' if args.initialization=='natural' else [int(x) for x in args.initialization.split(',')],
                 schedule=args.schedule,device=args.device,sentiment=not args.no_sentiment)
    if args.command in ('run', 'study') and args.device is not None:
        raw['device'] = args.device
    for key in ('repetitions', 'seed'):
        value = getattr(args, key, None)
        if value is not None:
            raw[key] = value
    if getattr(args, 'seed', None) is not None and raw.get('sampling'):
        raw['sampling'] = dict(raw['sampling'], seed=args.seed)
    config=resolve(raw)
    if getattr(args,'dry_run',False):
        from .prompts import templates,initial
        from .planning import configurations
        planned=list(configurations(config))
        print(json.dumps(dict(conversations=sum(len(c['scenarios']) for c in planned),configuration_count=len(planned),config=config,prompt_preview=initial(config['scenarios'][0],templates(),config['presentation_order'],config['agents'])),indent=2));return
    from .runner import execute
    print(execute(config,args.run_id,resume=args.command=='resume'))

if __name__=='__main__':main()
