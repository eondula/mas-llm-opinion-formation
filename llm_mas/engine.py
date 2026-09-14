"""Deterministic communication; runtime injected for testability."""
from copy import deepcopy
from contextlib import nullcontext
from .prompts import initial, for_condition


def conversation(config, scenario, texts, runtimes, emit=lambda **event: None, decision_context=None):
    texts = for_condition(texts, config, scenario)
    n = config['agents']
    names = [f'Agent {chr(65+i)}' if i < 26 else f'Agent {i+1}' for i in range(n)]
    models = config['models'] * n if len(config['models']) == 1 else config['models']
    histories = [initial(scenario, texts, config['presentation_order'], n, persona=i%2+1 if config.get('persona') else None, prompt_style=config.get('prompt_style','original'), agent_name=names[i] if config.get('prompt_style','original') != 'original' or config['mode'] == 'joint_task' else None, joint_task_brief=config.get('joint_task_brief') if config['mode'] == 'joint_task' else None) for i, _ in enumerate(names)]
    counts = [{1:0,2:0} for _ in names]
    messages, rows = [None]*n, []
    similarity = None
    for t in range(config['rounds']+1):
        old = list(messages)
        order = list(range(n))
        if config['schedule'] == 'reverse_sequential': order.reverse()
        for i in order:
            runtime = runtimes[models[i]]
            runtime.interaction_context = dict(scenario=scenario['id'], agent=names[i], round=t, stage='before')
            before = runtime.measure(histories[i], scenario, texts)
            deliveries = []
            if t and config['schedule'] != 'no_peer':
                incoming = old if config['schedule'] == 'synchronous' else messages
                for j in range(n):
                    if j != i:
                        deliveries.append(dict(speaker=names[j], receiver=names[i], message=incoming[j], round=t))
                        histories[i].append({'role':'user','content':texts['peer_message'].format(speaker=names[j], message=incoming[j])})
            generation_options = {}
            if config.get('decoding') == 'fixed_sampling':
                sampling = dict(config['sampling'])
                sampling['seed'] += t*n+i
                generation_options['sampling'] = sampling
            discussion_generation = None
            pre_discussion = None
            discuss_first = config.get('prompt_style') == 'discussion_then_decision' and t > 0
            if discuss_first:
                runtime.interaction_context['stage'] = 'pre_discussion'
                pre_discussion = runtime.measure(histories[i], scenario, texts)
                prompt = discussion_prompt(config, texts, scenario, deliveries, '')
                runtime.interaction_context['stage'] = 'discussion'
                discussion_generation = dict(runtime.generate(
                    histories[i]+[{'role':'user','content':prompt}],
                    config['max_argument_tokens'], **generation_options), instruction=prompt)
                # Temporary response is used for reassessment, then consolidated into
                # the single public message retained and delivered to peers.
                histories[i].append({'role':'assistant','content':discussion_generation['output']})
            forced = t == 0 and config['initialization'] != 'natural'
            if forced:
                strategy = config['initialization'][i]
                histories[i].append({'role':'user','content':texts['starting_assessment'].format(candidate=scenario[f'strategy_{strategy}']['candidate'])})
                decision = None
            else:
                runtime.interaction_context['stage'] = 'decision'
                with decision_context(t, names[i], runtime, histories[i]) if decision_context else nullcontext():
                    decision = before if t == 0 else runtime.measure(histories[i], scenario, texts)
                strategy = 1 if decision['margin'] >= 0 else 2
            decision_history = deepcopy(histories[i])
            selected = scenario[f'strategy_{strategy}']
            choice_status = 'forced' if forced else 'scored'
            runtime.interaction_context['stage'] = 'response'
            generation = None
            argument_selection = None
            argument_index = None
            autonomous = config.get('decision_mode') == 'generated' and not forced
            if autonomous:
                from .choice_output import parse, instruction, recovery, PARSER_VERSION
                prompt = instruction(texts, scenario)
                if discuss_first:
                    prompt += '\nReassess after your discussion above. You may retain or revise your strategy. Give a brief reason for your decision.'
                elif config.get('prompt_style', 'original') != 'original':
                    prompt = discussion_prompt(config, texts, scenario, deliveries, 'Choose your current strategy') + '\nReturn only a JSON object with strategy (integer 1 or 2) and explanation (your conversational response).'
                generation = runtime.generate(histories[i]+[{'role':'user','content':prompt}], config['max_argument_tokens'], **generation_options)
                generation = dict(generation, instruction=prompt)
                strategy, argument, choice_status = parse(generation['output'])
                generation.update(choice_parser_version=PARSER_VERSION, choice_recovery=recovery(generation['output']))
            elif discuss_first:
                argument = discussion_generation['output']
            elif config['argument_mode'] in ('generated', 'conversational'):
                prompt = texts['generate_argument'].format(candidate=selected['candidate'])
                if config['argument_mode'] == 'conversational':
                    reply_instruction = (
                        'Respond to the latest peer messages delivered above. Address a relevant reason or tradeoff raised by a peer rather than simply restating their conclusion.'
                        if deliveries else
                        'No peer message has been delivered. Give your own rationale for the selected position; do not imply that another agent has spoken.')
                    prompt = texts['conversational_argument'].format(candidate=selected['candidate'], reply_instruction=reply_instruction)
                generation = runtime.generate(histories[i] + [{'role':'user','content':prompt}], config['max_argument_tokens'], **generation_options)
                generation = dict(generation, instruction=prompt, argument_mode=config['argument_mode'])
                argument = generation['output']
            elif config['argument_mode'] == 'combined_reasoning':
                prompt = texts['stance_probe'].format(candidate_1=scenario['strategy_1']['candidate'], candidate_2=scenario['strategy_2']['candidate'])
                if config.get('prompt_style', 'original') != 'original':
                    prompt = discussion_prompt(config, texts, scenario, deliveries, selected['candidate'])
                prefix = selected['candidate'] + '\n'
                generation = runtime.generate(histories[i] + [{'role':'user','content':prompt}], config['max_argument_tokens'], continuation_prefix=prefix, **generation_options)
                generation = dict(generation, instruction=prompt, continuation_prefix=prefix, argument_mode='combined_reasoning')
                argument = generation['output']
            elif config['argument_mode'] == 'combined_fixed':
                argument_selection = runtime.select_combined_argument(histories[i], selected, scenario, texts)
                argument_index = argument_selection['index']
                argument = selected['arguments'][argument_index]
            elif config['argument_mode'] == 'agent_selected':
                argument_selection = runtime.select_argument(histories[i], selected, texts)
                argument_index = argument_selection['index']
                argument = selected['arguments'][argument_index]
            else:
                argument_index = counts[i][strategy] % len(selected['arguments'])
                argument = selected['arguments'][argument_index]
                counts[i][strategy] += 1
            message = generation['output'] if autonomous else selected['candidate']+'\n'+argument
            if discuss_first:
                histories[i].pop()  # Remove temporary discussion; retain it once below.
                if autonomous:
                    argument = discussion_generation['output']+'\n\nReassessment: '+argument
                    message = discussion_generation['output']+'\n\nReassessment:\n'+generation['output']
                else:
                    message = argument+'\n\nDecision: '+selected['candidate']
            messages[i] = message
            histories[i].append({'role':'assistant','content':message})
            runtime.interaction_context['stage'] = 'post'
            post = runtime.measure(histories[i], scenario, texts)
            row = dict(round=t, agent=names[i], model=models[i], strategy=strategy, forced=forced,
                       before=before, decision=decision, post=post, message=message, argument=argument, choice_status=choice_status,
                       generation=generation, argument_index=argument_index, argument_selection=argument_selection, deliveries=deliveries, decision_history=decision_history, history=deepcopy(histories[i]))
            if config.get('prompt_style') == 'discussion_then_decision':
                row.update(prompt_style='discussion_then_decision',
                           decision_timing='after_discussion' if discuss_first else 'initialization',
                           pre_discussion=pre_discussion, discussion_generation=discussion_generation)
            rows.append(row)
            emit(kind='response', row=row, completed=len(rows), total=(config['rounds']+1)*n)
        if config.get('stopping') == 'agent_closure' and t >= 1:
            from .closure import check
            termination = check(config, texts, runtimes, models, names, histories, rows[-n:], t)
            for i, name in enumerate(names):
                row = next(r for r in rows[-n:] if r['agent'] == name)
                row['termination'] = termination
                row['history_after_closure'] = deepcopy(histories[i])
            emit(kind='termination_check', **termination,
                 completed=len(rows), total=(config['rounds']+1)*n)
            if termination['reason']:
                break
        if config.get('stopping') in ('exact_repetition', 'cosine_similarity', 'combined_stability', 'sign_stability') and t >= 1:
            details = {}
            if config['stopping'] == 'cosine_similarity' and t < config.get('minimum_rounds', 2):
                matched = []
            elif config['stopping'] in ('cosine_similarity', 'combined_stability', 'sign_stability'):
                if similarity is None:
                    from .message_similarity import MessageSimilarity
                    similarity = MessageSimilarity()
                scores = similarity.compare(messages, old)
                threshold = config.get('similarity_threshold', 0.8)
                matched = [name for name, score in zip(names, scores) if score >= threshold]
                details = dict(similarities=dict(zip(names, scores)), threshold=threshold,
                               encoder=similarity.encoder, encoder_revision=similarity.revision,
                               input='complete_public_message', pooling='length-weighted chunk mean, L2 normalized')
            else:
                matched = [name for i, name in enumerate(names) if messages[i] == old[i]]
            if config['stopping'] in ('combined_stability', 'sign_stability'):
                previous = {r['agent']:r for r in rows[-2*n:-n]}
                current = {r['agent']:r for r in rows[-n:]}
                deltas = {name:abs(current[name]['post']['margin']-previous[name]['post']['margin']) for name in names}
                unchanged = {name:current[name]['strategy'] is not None and previous[name]['strategy'] is not None and current[name]['strategy']==previous[name]['strategy'] for name in names}
                details.update(margin_changes=deltas, strategy_unchanged=unchanged)
                if config['stopping'] == 'sign_stability':
                    signs = {name:(current[name]['post']['margin'] >= 0) == (previous[name]['post']['margin'] >= 0) for name in names}
                    matched = [name for name in matched if unchanged[name] and signs[name]]
                    details.update(margin_sign_unchanged=signs, zero_convention='nonnegative_favors_strategy_1')
                else:
                    tolerance = config.get('margin_tolerance', 0.1)
                    matched = [name for name in matched if unchanged[name] and deltas[name] <= tolerance]
                    details.update(margin_tolerance=tolerance)
            minimum = config.get('minimum_rounds', 2) if config['stopping'] == 'cosine_similarity' else 1
            reason = (('system_detected_stability' if config['stopping'] in ('combined_stability', 'sign_stability') else 'system_detected_repetition') if t >= minimum and len(matched) == n
                      else 'max_rounds' if t == config['rounds'] else None)
            termination = dict(round=t, previous_round=t-1, matched_agents=matched,
                               reason=reason, policy=config['stopping'], **details)
            for row in rows[-n:]:
                row['termination'] = termination
            emit(kind='termination_check', **termination,
                 completed=len(rows), total=(config['rounds']+1)*n)
            if reason and not config.get('observe_stopping', False):
                break
    if config['mode'] == 'joint_task':
        from .joint_task import decision_sheet
        rows[-1]['joint_outcome'] = decision_sheet(rows, scenario, n, config['schedule'])
    return rows


def discussion_prompt(config, texts, scenario, deliveries, candidate):
    turn = ('Respond to the peer messages delivered above.' if deliveries else
            'This is initialization: no peer has spoken. Give your initial assessment or identify a question for later discussion; do not imply a peer has made a claim.')
    pools = '\n\n'.join('STRATEGY '+str(k)+'\n'+scenario[f'strategy_{k}']['candidate']+'\n'+'\n'.join(scenario[f'strategy_{k}']['arguments']) for k in (1,2))
    if config['prompt_style'] == 'topic_discussion':
        keys = (2,1) if config.get('reverse_candidate_blocks') else (1,2)
        pools = '\n\n'.join('STRATEGY '+str(k)+'\n'+scenario[f'strategy_{k}']['candidate'] for k in keys)
    style = config['prompt_style']
    if style == 'discussion_then_decision':
        if candidate:  # Initialization keeps the established choice-first behavior.
            style = 'open_discussion'
        else:
            turn = ('Respond to the peer messages delivered above.' if deliveries else
                    'No peer message has been delivered. Reconsider your own earlier reasoning; do not invent peer statements.')
        if config.get('reverse_candidate_blocks'):
            pools = '\n\n'.join('STRATEGY '+str(k)+'\n'+scenario[f'strategy_{k}']['candidate']+'\n'+'\n'.join(scenario[f'strategy_{k}']['arguments']) for k in (2,1))
    return pools+'\n\n'+texts[style].format(candidate=candidate,turn_instruction=turn)
