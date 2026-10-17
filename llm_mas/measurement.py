"""Preserved full-candidate stance measurement (candidate-log-likelihood-v1).

The model's native log-softmax arithmetic remains in Runtime.likelihood.
This probe operates on a temporary list and never enters the live history.
"""
import math


def measure(runtime, history, scenario, texts):
    from .tasks import candidates, is_question
    c1, c2 = candidates(scenario)
    probe = texts['qa_probe' if is_question(scenario) else 'stance_probe'].format(candidate_1=c1, candidate_2=c2)
    messages = history + [{'role':'user','content':probe}]
    l1, t1, rendered = runtime.likelihood(messages, c1)
    l2, t2, _ = runtime.likelihood(messages, c2)
    margin = l1-l2
    return dict(margin=margin, stance=math.tanh(margin/2), likelihoods=[l1,l2], token_log_probs=[t1,t2],
                candidate_tokens=[runtime.tokenizer.convert_ids_to_tokens(runtime.encode_rendered(rendered+c)['input_ids'][len(runtime.encode_rendered(rendered)['input_ids']):]) for c in (c1,c2)],
                token_counts=[len(t1),len(t2)], normalized_margin=l1/len(t1)-l2/len(t2) if t1 and t2 else None,
                rendered_prompt=rendered, probe=probe)

