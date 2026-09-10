"""Controlled measurements over saved conversation histories.

These probes never enter the live conversations. No claim of independent
validation is made from the public choice constructed by the response policy.
"""
from copy import deepcopy
from itertools import product
import json
from .prompts import initial
from .runner import write


def run(path, config, texts, runtimes, progress):
    modules = [m for m in config.get('research', []) if m != 'prediction']
    unknown = set(modules)-{'validation','history','replay','mechanistic','heads','source_spans','continuation','sufficiency'}
    if unknown: raise ValueError(f'Unsupported research modules: {sorted(unknown)}')
    for module in modules:
        target=path/'measurements'/f'{module}.json'
        if target.exists():continue
        if module in ('replay','mechanistic','heads','source_spans','continuation','sufficiency'):
            results=[]
            for source in sorted((path/'conversations').glob('*.json')):
                record=json.loads(source.read_text());scenario=record['scenario']
                eligible=[r for r in record['rows'] if r['round']>0 and r.get('deliveries')]
                # Select by chronological order, not by patch outcome.
                for row in eligible[:config.get('research_events_per_conversation',2)]:
                    runtime=runtimes[row['model']]
                    history=row['decision_history'];before=history[:-len(row['deliveries'])]
                    if module=='sufficiency':
                        same=[r for r in record['rows'] if r['model']==row['model'] and r['decision'] and r['decision_history']!=history and abs(r['decision']['margin']-row['decision']['margin'])<=config.get('stance_match_tolerance',0.1)]
                        if not same:
                            results.append(dict(task=source.stem,agent=row['agent'],round=row['round'],status='no_matched_history'));continue
                        other=min(same,key=lambda r:abs(r['decision']['margin']-row['decision']['margin']))
                        message={'role':'user','content':texts['peer_message'].format(speaker='Agent B',message=scenario['strategy_1']['candidate']+'\n'+scenario['strategy_1']['arguments'][0])}
                        a=runtime.measure(history+[message],scenario,texts);b=runtime.measure(other['decision_history']+[message],scenario,texts)
                        results.append(dict(task=source.stem,agent=row['agent'],round=row['round'],status='matched',before=[row['decision']['margin'],other['decision']['margin']],after=[a['margin'],b['margin']],message=message,histories=[history,other['decision_history']]))
                    elif module in ('heads','source_spans'):
                        from .interventions import head_sweep,source_span_sweep
                        function=head_sweep if module=='heads' else source_span_sweep
                        results.append(dict(task=source.stem,agent=row['agent'],round=row['round'],model=row['model'],effects=function(runtime,before,history,scenario,texts,config.get('patch_layers'))))
                    elif module=='continuation':
                        from .continuation import evaluate
                        results.append(evaluate(record,row,before,scenario,texts,runtimes,source.stem,config.get('continuation_layer',-1)))
                    elif module=='mechanistic':
                        from .interventions import sweep
                        effects=sweep(runtime,before,history,scenario,texts,config.get('patch_layers'))
                        results.append(dict(task=source.stem,agent=row['agent'],round=row['round'],model=row['model'],effects=effects,
                            scope='Exploratory final-prompt-position residual/attention/MLP substitution; no downstream continuation'))
                    else:
                        variants={'original':history,'reverse_delivery':before+list(reversed(history[len(before):])),
                                  'unattributed':before+[{'role':'user','content':d['message']} for d in row['deliveries']],
                                  'no_self':[m for m in history if m['role']!='assistant']}
                        scores={name:runtime.measure(h,scenario,texts) for name,h in variants.items()}
                        results.append(dict(task=source.stem,agent=row['agent'],round=row['round'],model=row['model'],scores=scores,histories=variants))
                    progress(module,row['model'],scenario['id'])
            write(target,results)
            continue
        results=[]
        for model in list(runtimes):
            runtime=runtimes[model]
            for scenario in config['scenarios']:
                history=initial(scenario,texts,config['presentation_order'],config['agents'])
                if module=='validation':
                    from .validation import evaluate
                    result=evaluate(runtime,history,scenario,texts,config['max_argument_tokens'])
                    opposite='strategy_2_first' if config['presentation_order']=='strategy_1_first' else 'strategy_1_first'
                    result['reversed_presentation']=runtime.measure(initial(scenario,texts,opposite,config['agents']),scenario,texts)
                    results.append(dict(model=model,scenario=scenario['id'],**result))
                else:
                    for own,peer in product((1,2),repeat=2):
                        selftext=scenario[f'strategy_{own}']['candidate']+'\n'+scenario[f'strategy_{own}']['arguments'][0]
                        peertext=scenario[f'strategy_{peer}']['candidate']+'\n'+scenario[f'strategy_{peer}']['arguments'][0]
                        h=history+[{'role':'assistant','content':selftext}]
                        before=runtime.measure(h,scenario,texts)
                        received=h+[{'role':'user','content':texts['peer_message'].format(speaker='Agent B',message=peertext)}]
                        after=runtime.measure(received,scenario,texts)
                        results.append(dict(model=model,scenario=scenario['id'],self_strategy=own,peer_strategy=peer,
                                            before=before,after=after,delta=after['margin']-before['margin'],history=received))
                progress(module,model,scenario['id'])
        write(target,results)
