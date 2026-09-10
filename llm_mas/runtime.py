import math
import weakref

class Runtime:
    instances=weakref.WeakSet()
    def __init__(self, name, device='auto'):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype='auto').eval()
        self.instances.add(self)
        self.activate()
        self.name = name

    def activate(self):
        for other in list(self.instances):
            if other is not self and next(other.model.parameters()).device.type!='cpu':other.model.to('cpu')
        self.model.to(self.device)

    def inputs(self, messages, candidate=''):
        self.activate()
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(rendered + candidate, return_tensors='pt').to(self.device)
        limit = getattr(self.model.config, 'max_position_embeddings', None)
        if limit and encoded.input_ids.shape[1] > limit: raise ValueError('Context exceeds model limit; history was not truncated')
        return rendered, encoded

    def likelihood(self, messages, candidate):
        rendered, encoded = self.inputs(messages, candidate)
        prefix=self.tokenizer(rendered)['input_ids']
        start = len(prefix)
        if encoded.input_ids[0,:start].tolist()!=prefix:raise ValueError('Candidate changes prompt token boundary')
        with self.torch.inference_mode():
            lp = self.torch.log_softmax(self.model(**encoded).logits, dim=-1)
        values = [lp[0, i-1, encoded.input_ids[0,i]].item() for i in range(start, encoded.input_ids.shape[1])]
        return sum(values), values, rendered

    def measure(self, history, scenario, texts):
        c1, c2 = [scenario[k]['candidate'] for k in ('strategy_1','strategy_2')]
        probe = texts['stance_probe'].format(candidate_1=c1, candidate_2=c2)
        messages = history + [{'role':'user','content':probe}]
        l1, t1, rendered = self.likelihood(messages, c1)
        l2, t2, _ = self.likelihood(messages, c2)
        margin = l1-l2
        return dict(margin=margin, stance=math.tanh(margin/2), likelihoods=[l1,l2], token_log_probs=[t1,t2],
                    candidate_tokens=[self.tokenizer.convert_ids_to_tokens(self.tokenizer(rendered+c)['input_ids'][len(self.tokenizer(rendered)['input_ids']):]) for c in (c1,c2)],
                    token_counts=[len(t1),len(t2)], normalized_margin=l1/len(t1)-l2/len(t2) if t1 and t2 else None,
                    rendered_prompt=rendered, probe=probe)

    def generate(self, messages, max_tokens):
        rendered, inputs = self.inputs(messages)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, num_beams=1,
                                         pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        if not text: raise ValueError('Empty generated argument')
        return dict(output=text, rendered_prompt=rendered, max_new_tokens=max_tokens, do_sample=False, num_beams=1)
