"""One conversation scheduler for independent and deliberating agents.

Response policies preserve historical prompting, parsing and scoring semantics.
"""
from copy import deepcopy
from .response_policy import LegacyResponsePolicy, discussion_prompt


def names_for(n):
    return [f'Agent {chr(65+i)}' if i < 26 else f'Agent {i+1}' for i in range(n)]


def conversation(config, scenario, texts, runtimes, emit=lambda **event: None, decision_context=None, initial_rows=None):
    n = config['agents']
    names = names_for(n)
    models = config['models'] * n if len(config['models']) == 1 else config['models']
    exploration = config['mode'] == 'scenario_exploration'
    from .tasks import is_question
    if is_question(scenario):
        from .question_policy import QuestionResponsePolicy
        policy = QuestionResponsePolicy(config, scenario, texts, names, models)
    elif exploration:
        from .exploration import ExplorationResponsePolicy
        policy = ExplorationResponsePolicy(config, scenario, texts, names, models)
    else:
        policy = LegacyResponsePolicy(config, scenario, texts, names, models)
    texts = policy.texts
    histories = [policy.initial(i) for i in range(n)]
    messages, rows, latest = [None] * n, [], [None] * n
    if initial_rows is not None:
        if len(initial_rows)!=n or any(r['round']!=0 or r['agent']!=names[i] or r.get('question_id')!=scenario['id'] or r['model']!=models[i] for i,r in enumerate(initial_rows)):
            raise ValueError('Invalid shared initialization snapshot')
        rows=deepcopy(initial_rows)
        histories=[deepcopy(r['history']) for r in rows]
        messages=[r['message'] for r in rows];latest=list(rows)
        for i,row in enumerate(rows):emit(kind='response',row=deepcopy(row),completed=i+1,total=(config['rounds']+1)*n,generation_reused=True)
    similarity = None
    for t in range(1 if initial_rows is not None else 0,config['rounds'] + 1):
        old = list(messages)
        old_rows = list(latest)
        order = list(range(n))
        if config['schedule'] == 'reverse_sequential':
            order.reverse()
        for i in order:
            runtime = runtimes[models[i]]
            runtime.interaction_context = dict(scenario=scenario['id'], agent=names[i], round=t,
                                               stage='exploration_response' if exploration else 'before')
            before = policy.before(runtime, histories[i])
            deliveries = []
            if t and config['schedule'] != 'no_peer':
                incoming = old_rows if config['schedule'] == 'synchronous' else latest
                for j in range(n):
                    if j == i:
                        continue
                    source = incoming[j]
                    deliveries.append(dict(speaker=names[j], receiver=names[i], message=source['message'],
                                           round=t, source_id=source['response_id'], source_round=source['round']))
                    histories[i].append(dict(role='user', content=texts['peer_message'].format(
                        speaker=names[j], message=source['message'])))
            row = policy.respond(runtime, histories[i], t, i, deliveries, before, decision_context)
            row.setdefault('response_id', f'r{t}_a{i}')
            messages[i] = row['message']
            latest[i] = row
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
    if exploration:
        rows[-1]['termination'] = dict(reason='max_rounds', round=config['rounds'], policy='fixed')
    if config['mode'] == 'joint_task':
        from .joint_task import decision_sheet
        rows[-1]['joint_outcome'] = decision_sheet(rows, scenario, n, config['schedule'])
    return rows
