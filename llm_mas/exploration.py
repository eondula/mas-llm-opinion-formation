"""Generated public choices with an optional, non-interacting diagnostic."""
import json
import math
from copy import deepcopy


def names_for(n):
    return [f'Agent {chr(65+i)}' if i < 26 else f'Agent {i+1}' for i in range(n)]


def references(scenario):
    return {f'S{k}_A{j+1}': (k, text) for k in (1, 2)
            for j, text in enumerate(scenario[f'strategy_{k}']['arguments'])}


def assigned_evidence(config, scenario, i):
    access = config.get('information_access', 'none')
    cards = scenario.get('evidence_cards', [])
    if access == 'none': return []
    if access == 'shared': return deepcopy(cards)
    # Round-robin partitions preserve the group evidence set; extra agents may have no card.
    return deepcopy([card for j, card in enumerate(cards)
                     if (j+config.get('evidence_rotation', 0)) % config['agents'] == i])


def initial(config, scenario, texts, i):
    names = names_for(config['agents'])
    persona = ''
    if config.get('persona'):
        role = scenario['persona_roles'][f'strategy_{(i+config.get('persona_rotation', 0))%2+1}']
        persona = 'Your role: '+role['role_name']+'.\n'+role['priority']
    system = texts['exploration_system'].format(agent_name=names[i], peers=', '.join(n for n in names if n != names[i]), persona=persona)
    context = scenario['title']+'\n\n'+scenario.get('context', '')
    for k in (1, 2):
        option = scenario[f'strategy_{k}']
        context += f'\n\nStrategy {k}: '+option['candidate']+'\n'+option['property_1']+'\n'+option['property_2']
    if config.get('information_access', 'none') != 'none':
        cards = assigned_evidence(config, scenario, i)
        context += '\n\nEvidence available to you:\n' + ('\n'.join(card['id']+': '+card['text'] for card in cards) or 'No additional evidence card was assigned to you.')
        context += '\nThese are task facts. Share relevant evidence in your explanation. Do not assume access to other evidence until it is delivered in a peer message.'
    if config['argument_source'] == 'supplied':
        context += '\n\nAvailable arguments:\n'+'\n'.join(f'{key}: {text}' for key, (_, text) in references(scenario).items())
    return [dict(role='system', content=system), dict(role='user', content=context)]


def response_prompt(config, texts, t, i, has_peers):
    turn = ('Address a specific point in the delivered peer messages and name its author.' if has_peers else
            'No peer message has been delivered. Give your own assessment; do not imply a peer has spoken.')
    assignment = (f'For this initial response, select Strategy {config["initialization"][i]}. This assignment applies only at initialization.'
                  if t == 0 and config['initialization'] != 'natural' else '')
    if config['argument_source'] == 'own':
        source = 'Explain the main reason for your choice in a short paragraph. If a peer point affected your assessment, explain how; otherwise explain why you retain it.'
        schema = 'Use this schema: {"strategy": 1 or 2, "explanation": "your reason"}.'
    else:
        source = 'Select exactly one supplied argument belonging to your selected strategy. Choose the argument most relevant to your assessment or the delivered peer point. Do not rewrite it.'
        schema = 'Use this schema: {"strategy": 1 or 2, "argument_id": "the supplied ID, such as S1_A3"}.'
    if config.get('exploration_setup', 'standard') == 'deliberation':
        if has_peers:
            turn = 'Identify one specific claim in the delivered peer messages that matters to the decision and name its author. Explain whether it changes your assessment and why.'
        return texts['exploration_deliberation'].format(turn=turn, assignment=assignment, schema=schema)
    return texts['exploration_response'].format(turn=turn, assignment=assignment, source=source, schema=schema)


def parse(output, source, scenario):
    try:
        value = json.loads(output)
    except (ValueError, TypeError):
        return None, '', None, 'invalid_json'
    if not isinstance(value, dict) or type(value.get('strategy')) is not int or value['strategy'] not in (1, 2):
        return None, '', None, 'invalid_schema'
    strategy = value['strategy']
    if source == 'own' and set(value) == {'strategy', 'explanation'} and isinstance(value['explanation'], str) and value['explanation'].strip():
        return strategy, value['explanation'], None, 'valid'
    if source == 'supplied' and set(value) == {'strategy', 'argument_id'} and isinstance(value['argument_id'], str):
        ref = references(scenario).get(value['argument_id'])
        if ref and ref[0] == strategy:
            return strategy, ref[1], value['argument_id'], 'valid'
    return None, '', None, 'invalid_schema'


def diagnostic(runtime, history, scenario, texts):
    candidates = [scenario[f'strategy_{k}']['candidate'] for k in (1, 2)]
    probe = texts['exploration_probe'].format(candidate_1=candidates[0], candidate_2=candidates[1])
    measured = []
    for candidate in candidates:
        rendered, encoded = runtime.inputs(history+[dict(role='user', content=probe)], candidate)
        prefix = runtime.encode_rendered(rendered)['input_ids']
        start = len(prefix)
        if encoded.input_ids[0, :start].tolist() != prefix:
            raise ValueError('Candidate changes prompt token boundary')
        torch = runtime.torch
        with torch.inference_mode():
            output = runtime.model(**encoded, use_cache=False)
            logits = output.logits[:, start-1:encoded.input_ids.shape[1]-1, :].clone()
            del output
            lp = torch.log_softmax(logits.float(), dim=-1)
            ids = encoded.input_ids[:, start:]
            scores = lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)[0]
            total = scores.sum(dtype=torch.float32).item()
        if not math.isfinite(total):
            raise ValueError('Nonfinite diagnostic score')
        measured.append(dict(score=total, token_scores=scores.tolist(), token_ids=ids[0].tolist(),
                             tokens=runtime.tokenizer.convert_ids_to_tokens(ids[0].tolist()),
                             logits_dtype=str(logits.dtype), log_softmax_dtype=str(lp.dtype),
                             gathered_dtype=str(scores.dtype), accumulation_dtype='torch.float32'))
    margin = measured[0]['score']-measured[1]['score']
    return dict(status='ok', stage='post_response', margin=margin, stance=math.tanh(margin/2),
                strategy=1 if margin > 0 else 2 if margin < 0 else None, tie=margin == 0,
                candidates=measured, probe=probe, rendered_prompt=rendered)


class ExplorationResponsePolicy:
    def __init__(self, config, scenario, texts, names, models):
        self.config, self.scenario, self.texts = config, scenario, texts
        self.names, self.models = names, models

    def initial(self, i):
        return initial(self.config, self.scenario, self.texts, i)

    def before(self, runtime, history):
        return None

    def respond(self, runtime, history, t, i, deliveries, before, decision_context=None):
        config, scenario, texts = self.config, self.scenario, self.texts
        names, models = self.names, self.models
        prompt = response_prompt(config, texts, t, i, bool(deliveries))
        generation = dict(runtime.generate(history+[dict(role='user', content=prompt)], config['max_argument_tokens']), instruction=prompt)
        strategy, argument, argument_id, validity = parse(generation['output'], config['argument_source'], scenario)
        message = scenario[f'strategy_{strategy}']['candidate']+'\n'+argument if validity == 'valid' else generation['output']
        history.append(dict(role='assistant', content=message))
        post = dict(status='disabled', stance=None, margin=None)
        if config['stance_diagnostic']:
            runtime.interaction_context['stage'] = 'exploration_diagnostic'
            try:
                post = diagnostic(runtime, history, scenario, texts)
            except ValueError as error:
                post = dict(status='error', stage='post_response', stance=None, margin=None, error=str(error))
        forced = t == 0 and config['initialization'] != 'natural'
        assignment = config['initialization'][i] if forced else None
        row = dict(response_id=f'r{t}_a{i}', round=t, agent=names[i], model=models[i], strategy=strategy,
                   forced=forced, forced_assignment=assignment, forced_compliance=(strategy==assignment if strategy is not None else None) if forced else None,
                   information_access=config.get('information_access', 'none'), initial_evidence=assigned_evidence(config, scenario, i),
                   argument_source=config['argument_source'], argument=argument, argument_id=argument_id,
                   choice_status=validity, message=message, generation=generation, deliveries=deliveries,
                   post=post, diagnostic=post, history=deepcopy(history))
        return row


def conversation(config, scenario, texts, runtimes, emit=lambda **event: None):
    """Compatibility entry point; all schedules execute in the shared engine."""
    from .engine import conversation as execute
    return execute(config, scenario, texts, runtimes, emit)
