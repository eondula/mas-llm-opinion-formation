"""API agents for the large-model comparison: Claude (Anthropic SDK), OpenAI (OpenAI SDK), and an offline mock.

Each agent receives the same chat history the local pipeline builds (system prompt, then
alternating user/assistant turns) and returns the generated text plus request metadata.
No token logits are available, so no stance is computed.
"""
import time


def _merge_consecutive(messages):
    """Anthropic expects alternating roles; keep consecutive user turns as separate text blocks."""
    merged = []
    for m in messages:
        block = {'type': 'text', 'text': m['content']}
        if merged and merged[-1]['role'] == m['role']:
            merged[-1]['content'].append(block)
        else:
            merged.append({'role': m['role'], 'content': [block]})
    return merged


class ClaudeAgent:
    provider = 'anthropic'

    def __init__(self, model, effort, max_tokens, max_retries=5):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=max_retries)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def settings(self):
        return dict(provider=self.provider, model=self.model, effort=self.effort, max_tokens=self.max_tokens,
                    thinking='adaptive', sampling='not settable on current Claude models (temperature/top_p rejected)')

    def generate(self, system, messages):
        request = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                       messages=_merge_consecutive(messages), thinking={'type': 'adaptive'})
        if self.effort:
            request['output_config'] = {'effort': self.effort}
        start = time.time()
        response = self.client.messages.create(**request)
        text = ''.join(block.text for block in response.content if block.type == 'text').strip()
        usage = response.usage
        return dict(text=text, stop_reason=response.stop_reason, model_returned=response.model, response_id=response.id,
                    refusal=response.stop_reason == 'refusal',
                    truncated=response.stop_reason == 'max_tokens', latency_seconds=time.time()-start,
                    usage=dict(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens),
                    request_parameters={k: v for k, v in request.items() if k not in ('messages', 'system')})


class OpenAIAgent:
    provider = 'openai'

    def __init__(self, model, reasoning_effort, temperature, max_tokens, max_retries=5):
        import openai
        self.openai = openai
        self.client = openai.OpenAI(max_retries=max_retries)
        self.model, self.reasoning_effort, self.temperature, self.max_tokens = model, reasoning_effort, temperature, max_tokens
        # Parameters a model rejects are dropped once and recorded, so every later call is identical.
        self.dropped = []

    def settings(self):
        return dict(provider=self.provider, model=self.model, reasoning_effort=self.reasoning_effort,
                    temperature=self.temperature, max_completion_tokens=self.max_tokens, dropped_parameters=self.dropped)

    def generate(self, system, messages):
        request = dict(model=self.model, max_completion_tokens=self.max_tokens,
                       messages=[{'role': 'system', 'content': system}]+[dict(m) for m in messages])
        if self.temperature is not None and 'temperature' not in self.dropped:
            request['temperature'] = self.temperature
        if self.reasoning_effort and 'reasoning_effort' not in self.dropped:
            request['reasoning_effort'] = self.reasoning_effort
        start = time.time()
        while True:
            try:
                response = self.client.chat.completions.create(**request)
                break
            except self.openai.BadRequestError as error:
                text = str(error).lower()
                bad = next((p for p in ('temperature', 'reasoning_effort') if p in request and p in text), None)
                if bad is None:
                    raise
                print(f'  {self.model}: parameter {bad}={request[bad]!r} rejected; dropping it for this run')
                self.dropped.append(bad)
                request.pop(bad)
        choice = response.choices[0]
        usage = response.usage
        return dict(text=(choice.message.content or '').strip(), stop_reason=choice.finish_reason, model_returned=response.model,
                    response_id=response.id, refusal=bool(getattr(choice.message, 'refusal', None)),
                    truncated=choice.finish_reason == 'length', latency_seconds=time.time()-start,
                    usage=dict(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens) if usage else None,
                    system_fingerprint=getattr(response, 'system_fingerprint', None),
                    request_parameters={k: v for k, v in request.items() if k != 'messages'})


class MockAgent:
    """Offline stand-in with deterministic answers, for checking the pipeline without API keys or cost."""
    provider = 'mock'

    def __init__(self, model, lean):
        self.model, self.lean = model, lean

    def settings(self):
        return dict(provider=self.provider, model=self.model)

    def generate(self, system, messages):
        rounds = sum(m['role'] == 'assistant' for m in messages)
        peers = [m['content'] for m in messages if m['role'] == 'user' and m['content'].startswith('Message from')]
        answer = self.lean if rounds % 3 != 2 else ('No' if self.lean == 'Yes' else 'Yes')
        reply = f'Round {rounds}: {"responding to " + peers[-1].split(" (")[0][13:] if peers else "re-checking my reasoning"}; mock agent {self.model} considers the question.'
        return dict(text=f'{reply}\nAnswer: {answer}', stop_reason='end_turn', model_returned=self.model, response_id=None,
                    refusal=False, truncated=False, latency_seconds=0., usage=None, request_parameters={})
