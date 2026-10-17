"""Historical response policy, independent of message scheduling."""
from copy import deepcopy
from contextlib import nullcontext
from .prompts import initial, for_condition


class LegacyResponsePolicy:
    def __init__(self, config, scenario, texts, names, models):
        self.config, self.scenario = config, scenario
        self.texts = for_condition(texts, config, scenario)
        self.names, self.models = names, models
        self.counts = [{1: 0, 2: 0} for _ in names]

    def initial(self, i):
        config, scenario, texts = self.config, self.scenario, self.texts
        return initial(scenario, texts, config['presentation_order'], config['agents'],
                       persona=i % 2 + 1 if config.get('persona') else None,
                       prompt_style=config.get('prompt_style', 'original'),
                       agent_name=self.names[i] if config.get('prompt_style', 'original') != 'original' or config['mode'] == 'joint_task' else None,
                       joint_task_brief=config.get('joint_task_brief') if config['mode'] == 'joint_task' else None)

    def before(self, runtime, history):
        return runtime.measure(history, self.scenario, self.texts)

    def respond(self, runtime, history, t, i, deliveries, before, decision_context=None):
        config, scenario, texts = self.config, self.scenario, self.texts
        names, models, counts = self.names, self.models, self.counts
        n = config['agents']
        # Keep the established response construction and scoring order intact.
        generation_options = {}
        if config.get('decoding') == 'fixed_sampling':
            sampling = dict(config['sampling'])
            sampling['seed'] += config.get('seed_offset', 0) + t*n+i
            generation_options['sampling'] = sampling
        discussion_generation = None
        pre_discussion = None
        discuss_first = config.get('prompt_style') == 'discussion_then_decision' and t > 0
        if discuss_first:
            runtime.interaction_context['stage'] = 'pre_discussion'
            pre_discussion = runtime.measure(history, scenario, texts)
            prompt = discussion_prompt(config, texts, scenario, deliveries, '')
            runtime.interaction_context['stage'] = 'discussion'
            discussion_generation = dict(runtime.generate(
                history+[{'role':'user','content':prompt}],
                config['max_argument_tokens'], **generation_options), instruction=prompt)
            # Temporary response is used for reassessment, then consolidated into
            # the single public message retained and delivered to peers.
            history.append({'role':'assistant','content':discussion_generation['output']})
        forced = t == 0 and config['initialization'] != 'natural'
        if forced:
            strategy = config['initialization'][i]
            history.append({'role':'user','content':texts['starting_assessment'].format(candidate=scenario[f'strategy_{strategy}']['candidate'])})
            decision = None
        else:
            runtime.interaction_context['stage'] = 'decision'
            with decision_context(t, names[i], runtime, history) if decision_context else nullcontext():
                decision = before if t == 0 else runtime.measure(history, scenario, texts)
            strategy = 1 if decision['margin'] >= 0 else 2
        decision_history = deepcopy(history)
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
            generation = runtime.generate(history+[{'role':'user','content':prompt}], config['max_argument_tokens'], **generation_options)
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
            generation = runtime.generate(history + [{'role':'user','content':prompt}], config['max_argument_tokens'], **generation_options)
            generation = dict(generation, instruction=prompt, argument_mode=config['argument_mode'])
            argument = generation['output']
        elif config['argument_mode'] == 'combined_reasoning':
            prompt = texts['stance_probe'].format(candidate_1=scenario['strategy_1']['candidate'], candidate_2=scenario['strategy_2']['candidate'])
            if config.get('prompt_style', 'original') != 'original':
                prompt = discussion_prompt(config, texts, scenario, deliveries, selected['candidate'])
            prefix = selected['candidate'] + '\n'
            generation = runtime.generate(history + [{'role':'user','content':prompt}], config['max_argument_tokens'], continuation_prefix=prefix, **generation_options)
            generation = dict(generation, instruction=prompt, continuation_prefix=prefix, argument_mode='combined_reasoning')
            argument = generation['output']
        elif config['argument_mode'] == 'combined_fixed':
            argument_selection = runtime.select_combined_argument(history, selected, scenario, texts)
            argument_index = argument_selection['index']
            argument = selected['arguments'][argument_index]
        elif config['argument_mode'] == 'agent_selected':
            argument_selection = runtime.select_argument(history, selected, texts)
            argument_index = argument_selection['index']
            argument = selected['arguments'][argument_index]
        else:
            argument_index = counts[i][strategy] % len(selected['arguments'])
            argument = selected['arguments'][argument_index]
            counts[i][strategy] += 1
        message = generation['output'] if autonomous else selected['candidate']+'\n'+argument
        if discuss_first:
            history.pop()  # Remove temporary discussion; retain it once below.
            if autonomous:
                argument = discussion_generation['output']+'\n\nReassessment: '+argument
                message = discussion_generation['output']+'\n\nReassessment:\n'+generation['output']
            else:
                message = argument+'\n\nDecision: '+selected['candidate']
        history.append({'role':'assistant','content':message})
        runtime.interaction_context['stage'] = 'post'
        post = runtime.measure(history, scenario, texts)
        row = dict(response_id=f'r{t}_a{i}', round=t, agent=names[i], model=models[i], strategy=strategy, forced=forced,
                   before=before, decision=decision, post=post, message=message, argument=argument, choice_status=choice_status,
                   generation=generation, argument_index=argument_index, argument_selection=argument_selection, deliveries=deliveries, decision_history=decision_history, history=deepcopy(history))
        if config.get('prompt_style') == 'discussion_then_decision':
            row.update(prompt_style='discussion_then_decision',
                       decision_timing='after_discussion' if discuss_first else 'initialization',
                       pre_discussion=pre_discussion, discussion_generation=discussion_generation)
        return row

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
