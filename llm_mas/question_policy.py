"""Generated Yes/No judgments with independent, non-interacting stance probes."""
import json
from copy import deepcopy
from .prompts import question_initial

PARSER_VERSION = 'binary-answer-json-v1'


def parse(output):
    if not isinstance(output, str):
        return None, '', 'invalid_json'
    text = output.strip()
    lines = text.splitlines()
    if len(lines) > 2 and lines[0].strip().lower() in ('```json', '```') and lines[-1].strip() == '```':
        text = '\n'.join(lines[1:-1])
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None, '', 'invalid_json'
    if (not isinstance(value, dict) or set(value) != {'answer', 'explanation'}
            or not isinstance(value['answer'], str) or value['answer'].casefold() not in ('yes', 'no')
            or not isinstance(value['explanation'], str) or not value['explanation'].strip()):
        return None, '', 'invalid_schema'
    return value['answer'].capitalize(), value['explanation'].strip(), 'valid'


class QuestionResponsePolicy:
    def __init__(self, config, task, texts, names, models):
        self.config, self.task = config, task
        self.texts = dict(texts, peer_message=texts['qa_peer_message'])
        self.names, self.models = names, models

    def initial(self, i):
        return question_initial(self.task, self.texts, self.config['agents'], self.names[i], 'synchronous' if self.config.get('paired_control') else self.config['schedule'])

    def before(self, runtime, history):
        return runtime.measure(history, self.task, self.texts)

    def response_prompt(self, t, i, deliveries):
        forced = t == 0 and self.config['initialization'] != 'natural'
        if forced:
            answer = 'Yes' if self.config['initialization'][i] == 1 else 'No'
            turn = (f'No peer has spoken yet. Your assigned initial answer is {answer}. '
                    'Share your reasoning for this starting position. After this first turn, '
                    'you may keep or change your answer and argument in the conversation.')
        else:
            turn = ('Share your initial answer and reasoning. No peer has spoken yet.' if t == 0 else
                    'Continue the conversation, responding to your peers and referring to them by name where relevant. '
                    'You may keep or change your answer and argument based on the discussion.' if deliveries else
                    'Consider the question again and share your answer and reasoning. '
                    'You may keep or revise your earlier view. No peer message has been delivered.')
        if t>0 and self.config.get('paired_control'):
            turn='Continue considering the question and share your answer and reasoning. You may keep or change your view and refer to other agents by name when their messages are available.'
        template='qa_response_decision' if self.config.get('response_protocol')=='decision_time_v2' else 'qa_response'
        return self.texts[template].format(turn_instruction=turn)

    def respond(self, runtime, history, t, i, deliveries, before, decision_context=None):
        decision_history = deepcopy(history)
        runtime.interaction_context['stage'] = 'decision'
        decision = before if t == 0 else runtime.measure(history, self.task, self.texts)
        forced = t == 0 and self.config['initialization'] != 'natural'
        assigned = ('Yes' if self.config['initialization'][i] == 1 else 'No') if forced else None
        prompt = self.response_prompt(t, i, deliveries)
        options = {}
        if self.config['decoding'] == 'fixed_sampling':
            sampling = dict(self.config['sampling'])
            sampling['seed'] += self.config.get('seed_offset', 0) + t * self.config['agents'] + i
            options['sampling'] = sampling
        decision_time=None
        pre_response_probe=decision
        if self.config.get('response_protocol')=='decision_time_v2':
            result=runtime.generate_binary_response(history+[dict(role='user',content=prompt)],
                self.config['max_argument_tokens'],sampling=options.get('sampling'),assigned=assigned)
            answer,argument,status=result['answer'],result['argument'],'valid'
            generation=result['generation'];generation['instruction']=prompt
            decision_time=result['measurement'];decision=decision_time
        else:
            prefix = '{"answer": ' + json.dumps(assigned) + ', "explanation": ' if forced else ''
            if forced:
                options['continuation_prefix'] = prefix
            runtime.interaction_context['stage'] = 'response'
            generation = dict(runtime.generate(history + [dict(role='user', content=prompt)],
                                              self.config['max_argument_tokens'], **options),
                              instruction=prompt, choice_parser_version=PARSER_VERSION)
            generation['continuation_prefix'] = prefix
            generation['complete_output'] = prefix + generation['output']
            answer, argument, status = parse(generation['complete_output'])
            if forced and answer is not None and answer != assigned:
                answer, argument, status = None, '', 'invalid_forced_answer'
        message = answer + '\n' + argument if answer else generation['complete_output']
        history.append(dict(role='assistant', content=message))
        runtime.interaction_context['stage'] = 'post'
        post = runtime.measure(history, self.task, self.texts)
        return dict(response_id=f'r{t}_a{i}', question_id=self.task['id'], round=t,
                    agent=self.names[i], model=self.models[i], answer=answer,
                    strategy=1 if answer == 'Yes' else 2 if answer == 'No' else None,
                    choice_status=status, decision_time=decision_time, stance_source='decision_time' if decision_time else 'post_response_probe', pre_response_probe=pre_response_probe, forced=forced, assigned_initial_answer=assigned, before=before, decision=decision, post=post,
                    message=message, argument=argument, generation=generation,
                    deliveries=deliveries, decision_history=decision_history, history=deepcopy(history))
