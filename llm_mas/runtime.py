import math
import weakref

def select_device(torch, requested):
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if requested == 'auto':
        return 'cuda' if cuda else 'mps' if mps else 'cpu'
    if requested not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Unknown device: '+requested)
    if (requested == 'cuda' and not cuda) or (requested == 'mps' and not mps):
        raise ValueError(f'{requested.upper()} is unavailable. Use --device auto or select an available device.')
    return requested

class Runtime:
    instances=weakref.WeakSet()
    def __init__(self, name, device='auto', revision=None, tokenizer_revision=None, expected_dtype=None):
        print(f'Loading {name}: importing PyTorch', flush=True)
        import torch
        print(f'Loading {name}: importing Transformers', flush=True)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = select_device(torch, device)
        print(f'Loading {name}: tokenizer', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(name, **({'revision':tokenizer_revision or revision} if tokenizer_revision or revision else {}))
        self.tokenizer_revision=tokenizer_revision or revision
        print(f'Loading {name}: model weights', flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch,expected_dtype.removeprefix('torch.')) if expected_dtype else 'auto', **({'revision':revision} if revision else {})).eval()
        self.revision=revision
        if revision and getattr(self.model.config,'_commit_hash',None)!=revision:
            raise ValueError('Loaded model revision does not match pinned revision')
        if expected_dtype and str(next(self.model.parameters()).dtype)!=expected_dtype:
            raise ValueError('Loaded dtype does not match the frozen protocol')
        self.instances.add(self)
        print(f'Loading {name}: moving to {self.device}', flush=True)
        self.activate()
        self.name = name
        print(f'Ready: {name} on {self.device}', flush=True)

    def activate(self):
        for other in list(self.instances):
            if other is not self and next(other.model.parameters()).device.type!='cpu':other.model.to('cpu')
        self.model.to(self.device)

    def encode_rendered(self, text, **kwargs):
        # Use the special tokens supplied by each new checkpoint's chat template.
        # Preserve historical tokenization for existing study checkpoints.
        if getattr(self, 'name', '') in (
                'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'allenai/OLMo-2-0425-1B-Instruct', 'HuggingFaceTB/SmolLM3-3B', 'Qwen/Qwen3-4B-Instruct-2507', 'microsoft/Phi-4-mini-instruct'):
            kwargs['add_special_tokens'] = False
        return self.tokenizer(text, **kwargs)

    def check_context(self, input_tokens, reserve_tokens=0):
        limit = getattr(self.model.config, 'max_position_embeddings', None)
        if limit and input_tokens + reserve_tokens > limit:
            context = getattr(self, 'interaction_context', {})
            location = ', '.join(f'{key}={value}' for key, value in context.items())
            raise ValueError(
                f"Context exceeds model limit for {getattr(self, 'name', 'unknown model')}"
                f" ({location or 'standalone call'}): input={input_tokens} tokens, "
                f"reserved output={reserve_tokens}, limit={limit}. History was not truncated. "
                "Use a checkpoint with a larger context window, or start a new run with "
                "fewer rounds/agents or shorter messages. Lowering the output token budget "
                "does not shorten history already accumulated.")

    def render_chat(self, messages):
        options = dict(tokenize=False, add_generation_prompt=True)
        if getattr(self, 'name', '') == 'HuggingFaceTB/SmolLM3-3B':
            # The model card specifies /no_think in the system message. Keep
            # original histories intact and save the resulting rendered prompt.
            messages = [dict(message) for message in messages]
            if messages and messages[0]['role'] == 'system':
                messages[0]['content'] += '\n/no_think'
            else:
                messages.insert(0, dict(role='system', content='/no_think'))
            options['enable_thinking'] = False
        return self.tokenizer.apply_chat_template(messages, **options)

    def inputs(self, messages, candidate=''):
        self.activate()
        rendered = self.render_chat(messages)
        encoded = self.encode_rendered(rendered + candidate, return_tensors='pt').to(self.device)
        self.check_context(encoded.input_ids.shape[1])
        return rendered, encoded

    def likelihood(self, messages, candidate, continuation_prefix=''):
        rendered, encoded = self.inputs(messages, continuation_prefix + candidate)
        rendered += continuation_prefix
        prefix=self.encode_rendered(rendered)['input_ids']
        start = len(prefix)
        if encoded.input_ids[0,:start].tolist()!=prefix:raise ValueError('Candidate changes prompt token boundary')
        with self.torch.inference_mode():
            options = {'use_cache':False} if hasattr(self.model.config, 'use_cache') else {}
            output = self.model(**encoded, **options)
            # Only candidate prediction positions contribute to this likelihood.
            logits = output.logits[:, start-1:encoded.input_ids.shape[1]-1, :].clone()
            del output
            lp = self.torch.log_softmax(logits, dim=-1)
            ids = encoded.input_ids[:, start:]
            values = lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)[0].tolist()
        return sum(values), values, rendered

    def measure(self, history, scenario, texts):
        from .measurement import measure
        return measure(self, history, scenario, texts)

    def generate(self, messages, max_tokens, continuation_prefix='', sampling=None, allow_empty=False,
                 decoding=None, logits_processor=None, stopping_criteria=None):
        rendered, inputs = self.inputs(messages, continuation_prefix)
        self.check_context(inputs.input_ids.shape[1], reserve_tokens=max_tokens)
        options = dict(do_sample=bool(sampling), num_beams=1)
        # Explicit decoding settings override checkpoint generation_config defaults.
        if decoding:
            options.update(decoding)
        if sampling:
            self.torch.manual_seed(sampling['seed'])
            options.update({k:sampling[k] for k in ('temperature','top_p','top_k')})
        extra = {}
        if logits_processor:
            extra['logits_processor'] = logits_processor
        if stopping_criteria:
            extra['stopping_criteria'] = stopping_criteria
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, **options, **extra,
                                         pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        if not text and not allow_empty: raise ValueError('Empty generated argument')
        count=int(output.shape[1]-inputs.input_ids.shape[1])
        eos=int(output[0,-1])==self.tokenizer.eos_token_id
        return dict(output=text, generated_tokens=count, finish_reason='eos' if eos else 'length' if count>=max_tokens else 'other', rendered_prompt=rendered + continuation_prefix, continuation_prefix=continuation_prefix, max_new_tokens=max_tokens, **options, **({'seed':sampling['seed']} if sampling else {}))

    def generate_binary_response(self, messages, max_tokens, sampling=None, assigned=None):
        from .decision_generation import generate
        return generate(self,messages,max_tokens,sampling,assigned)

    def generate_reasoning_response(self, messages, max_tokens, own_messages, exempt, copy_block_ngram, sampling=None, assigned=None):
        from .decision_generation import generate_reasoning_first
        return generate_reasoning_first(self,messages,max_tokens,own_messages,exempt,copy_block_ngram,sampling,assigned)

    def select_argument(self, history, selected, texts):
        """Constrained deterministic choice among numbered argument entries."""
        arguments = selected['arguments']
        prompt = texts['select_argument'].format(
            candidate=selected['candidate'],
            arguments='\n'.join(f'{i+1}. {argument}' for i, argument in enumerate(arguments)))
        messages = history + [{'role': 'user', 'content': prompt}]
        labels = [str(i+1) for i in range(len(arguments))]
        scores = []
        rendered = None
        for label in labels:
            score, _, rendered = self.likelihood(messages, label)
            if not math.isfinite(score):
                raise ValueError('Non-finite argument selection score')
            scores.append(score)
        index = max(range(len(scores)), key=scores.__getitem__)
        return dict(index=index, label=labels[index], argument=arguments[index],
                    labels=labels, scores=scores, prompt=prompt, rendered_prompt=rendered,
                    method='number_likelihood_argmax', tie_break='first_in_list')

    def select_combined_argument(self, history, selected, scenario, texts):
        prompt = texts['stance_probe'].format(candidate_1=scenario['strategy_1']['candidate'],
                                            candidate_2=scenario['strategy_2']['candidate'])
        messages = history + [{'role':'user', 'content':prompt}]
        prefix = selected['candidate'] + '\n'
        scores, token_scores = [], []
        for argument in selected['arguments']:
            score, values, rendered = self.likelihood(messages, argument, continuation_prefix=prefix)
            if not math.isfinite(score):
                raise ValueError('Non-finite argument selection score')
            scores.append(score)
            token_scores.append(values)
        index = max(range(len(scores)), key=scores.__getitem__)
        return dict(index=index, argument=selected['arguments'][index], scores=scores,
                    token_log_probs=token_scores, token_counts=[len(v) for v in token_scores],
                    arguments=list(selected['arguments']), prompt=prompt, rendered_prompt=rendered,
                    continuation_prefix=prefix, method='argument_text_conditional_log_likelihood',
                    tie_break='first_in_list')
