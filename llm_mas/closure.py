"""Explicit proposal and unanimous confirmation, separate from strategy scoring."""
import json


def ask(runtime, history, template, instruction, tokens, stage):
    runtime.interaction_context['stage'] = stage
    prompt = template.format(instruction=instruction)
    generation = runtime.generate(history+[dict(role='user', content=prompt)], tokens)
    try:
        value = json.loads(generation['output'])
        valid = (isinstance(value, dict) and value.get('conversation_action') in ('continue', 'finish')
                 and isinstance(value.get('reason'), str) and bool(value['reason'].strip()))
    except (ValueError, TypeError):
        value, valid = {}, False
    return dict(action=value.get('conversation_action') if valid else None,
                reason=value.get('reason') if valid else None, valid=valid,
                generation=dict(generation, instruction=prompt))


def check(config, texts, runtimes, models, names, histories, rows, round_index):
    # Closure is an explicit extra communication phase, after a complete round.
    result = dict(round=round_index, policy='agent_closure', reason=None,
                  proposals=[], confirmations=[], proposal=None)
    if config['schedule'] == 'no_peer':
        result['unavailable'] = 'No peer control: closure proposals are not exchanged.'
    else:
        for i, name in enumerate(names):
            vote = ask(runtimes[models[i]], histories[i], texts['closure'],
                       'Do you propose ending the conversation after this round?',
                       config['max_argument_tokens'], 'closure_proposal')
            result['proposals'].append(dict(agent=name, **vote))
        proposer = next((v for v in result['proposals'] if v['action']=='finish'), None)
        if proposer:
            proposal = dict(id=f'closure_round_{round_index}', proposer=proposer['agent'],
                            reason=proposer['reason'],
                            positions=[dict(agent=r['agent'], strategy=r['strategy'], message=r['message']) for r in rows])
            result['proposal'] = proposal
            notice = 'Closure proposal and complete round record:\n'+json.dumps(proposal, ensure_ascii=False)
            # Every confirmation sees exactly this proposal, including the proposer.
            for i, name in enumerate(names):
                histories[i].append(dict(role='user', content=notice))
                vote = ask(runtimes[models[i]], histories[i], texts['closure'],
                           'Confirm this specific closure proposal after reviewing every recorded response above. '
                           'Return finish to accept closure, or continue to reopen discussion.',
                           config['max_argument_tokens'], 'closure_confirmation')
                result['confirmations'].append(dict(agent=name, **vote))
            accepted = all(v['action']=='finish' for v in result['confirmations'])
            valid_choices = all(r['strategy'] in (1,2) for r in rows)
            if accepted and valid_choices:
                result['reason'] = 'agents_accepted_closure'
            result['valid_choices'] = valid_choices
            notice = 'Closure responses:\n'+json.dumps([
                {k:v[k] for k in ('agent','action','reason','valid')} for v in result['confirmations']], ensure_ascii=False)
            notice += '\nConversation closed.' if result['reason'] else '\nClosure not accepted. Continue the discussion.'
            for history in histories:
                history.append(dict(role='user', content=notice))
    if result['reason'] is None and round_index == config['rounds']:
        result['reason'] = 'budget_exhausted'
    return result
