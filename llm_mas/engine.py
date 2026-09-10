"""Deterministic communication; runtime injected for testability."""
from copy import deepcopy
from contextlib import nullcontext
from .prompts import initial


def conversation(config, scenario, texts, runtimes, emit=lambda **event: None, decision_context=None):
    n = config['agents']
    names = [f'Agent {chr(65+i)}' if i < 26 else f'Agent {i+1}' for i in range(n)]
    models = config['models'] * n if len(config['models']) == 1 else config['models']
    histories = [initial(scenario, texts, config['presentation_order'], n) for _ in names]
    counts = [{1:0,2:0} for _ in names]
    messages, rows = [None]*n, []
    for t in range(config['rounds']+1):
        old = list(messages)
        order = list(range(n))
        if config['schedule'] == 'reverse_sequential': order.reverse()
        for i in order:
            runtime = runtimes[models[i]]
            before = runtime.measure(histories[i], scenario, texts)
            deliveries = []
            if t and config['schedule'] != 'no_peer':
                incoming = old if config['schedule'] == 'synchronous' else messages
                for j in range(n):
                    if j != i:
                        deliveries.append(dict(speaker=names[j], receiver=names[i], message=incoming[j], round=t))
                        histories[i].append({'role':'user','content':texts['peer_message'].format(speaker=names[j], message=incoming[j])})
            forced = t == 0 and config['initialization'] != 'natural'
            if forced:
                strategy = config['initialization'][i]
                histories[i].append({'role':'user','content':texts['starting_assessment'].format(candidate=scenario[f'strategy_{strategy}']['candidate'])})
                decision = None
            else:
                with decision_context(t, names[i], runtime, histories[i]) if decision_context else nullcontext():
                    decision = before if t == 0 else runtime.measure(histories[i], scenario, texts)
                strategy = 1 if decision['margin'] >= 0 else 2
            decision_history = deepcopy(histories[i])
            selected = scenario[f'strategy_{strategy}']
            generation = None
            if config['argument_mode'] == 'generated':
                generation = runtime.generate(histories[i] + [{'role':'user','content':texts['generate_argument'].format(candidate=selected['candidate'])}], config['max_argument_tokens'])
                argument = generation['output']
            else:
                argument = selected['arguments'][counts[i][strategy] % len(selected['arguments'])]
                counts[i][strategy] += 1
            message = selected['candidate']+'\n'+argument
            messages[i] = message
            histories[i].append({'role':'assistant','content':message})
            post = runtime.measure(histories[i], scenario, texts)
            row = dict(round=t, agent=names[i], model=models[i], strategy=strategy, forced=forced,
                       before=before, decision=decision, post=post, message=message, argument=argument,
                       generation=generation, deliveries=deliveries, decision_history=decision_history, history=deepcopy(histories[i]))
            rows.append(row)
            emit(kind='response', row=row, completed=len(rows), total=(config['rounds']+1)*n)
    return rows
