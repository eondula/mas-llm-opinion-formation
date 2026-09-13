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
    def __init__(self, name, device='auto'):
        print(f'Loading {name}: importing PyTorch', flush=True)
        import torch
        print(f'Loading {name}: importing Transformers', flush=True)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = select_device(torch, device)
        print(f'Loading {name}: tokenizer', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        print(f'Loading {name}: model weights', flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype='auto').eval()
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
                'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'allenai/OLMo-2-0425-1B-Instruct'):
            kwargs['add_special_tokens'] = False
        return self.tokenizer(text, **kwargs)

    def inputs(self, messages, candidate=''):
        self.activate()
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.encode_rendered(rendered + candidate, return_tensors='pt').to(self.device)
        limit = getattr(self.model.config, 'max_position_embeddings', None)
        if limit and encoded.input_ids.shape[1] > limit: raise ValueError('Context exceeds model limit; history was not truncated')
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
        c1, c2 = [scenario[k]['candidate'] for k in ('strategy_1','strategy_2')]
        probe = texts['stance_probe'].format(candidate_1=c1, candidate_2=c2)
        messages = history + [{'role':'user','content':probe}]
        l1, t1, rendered = self.likelihood(messages, c1)
        l2, t2, _ = self.likelihood(messages, c2)
        margin = l1-l2
        return dict(margin=margin, stance=math.tanh(margin/2), likelihoods=[l1,l2], token_log_probs=[t1,t2],
                    candidate_tokens=[self.tokenizer.convert_ids_to_tokens(self.encode_rendered(rendered+c)['input_ids'][len(self.encode_rendered(rendered)['input_ids']):]) for c in (c1,c2)],
                    token_counts=[len(t1),len(t2)], normalized_margin=l1/len(t1)-l2/len(t2) if t1 and t2 else None,
                    rendered_prompt=rendered, probe=probe)

    def generate(self, messages, max_tokens, continuation_prefix='', sampling=None):
        rendered, inputs = self.inputs(messages, continuation_prefix)
        options = dict(do_sample=bool(sampling), num_beams=1)
        if sampling:
            self.torch.manual_seed(sampling['seed'])
            options.update({k:sampling[k] for k in ('temperature','top_p','top_k')})
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, **options,
                                         pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        if not text: raise ValueError('Empty generated argument')
        return dict(output=text, generated_tokens=int(output.shape[1]-inputs.input_ids.shape[1]), rendered_prompt=rendered + continuation_prefix, continuation_prefix=continuation_prefix, max_new_tokens=max_tokens, **options, **({'seed':sampling['seed']} if sampling else {}))

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
