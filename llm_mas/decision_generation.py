"""Capture raw candidate logits at the constrained public-answer generation step."""
import json
import math

VERSION='answer-token-logits-v1'
PARSER='application-serialized-answer-v2'


def generate(runtime,messages,max_tokens,sampling=None,assigned=None):
    if max_tokens<2:raise ValueError('Decision-time responses require at least two tokens')
    torch=runtime.torch
    rendered,inputs=runtime.inputs(messages)
    runtime.check_context(inputs.input_ids.shape[1],reserve_tokens=max_tokens)
    prefix=runtime.encode_rendered(rendered)['input_ids']
    ids=[]
    for answer in ('Yes','No'):
        full=runtime.encode_rendered(rendered+answer)['input_ids']
        if full[:len(prefix)]!=prefix or len(full)!=len(prefix)+1:
            raise ValueError('Decision-time Yes/No candidates must each be one token at the exact response boundary; checkpoint unsupported')
        ids.append(full[-1])
    if len(set(ids))!=2:raise ValueError('Yes and No must have distinct token IDs')
    if assigned not in (None,'Yes','No'):raise ValueError('Unknown forced answer')
    if runtime.tokenizer.eos_token_id in ids:raise ValueError('Answer token cannot be EOS')
    allowed=[ids[0 if assigned=='Yes' else 1]] if assigned else ids
    class AnswerOnly:
        def __call__(self,input_ids,scores):
            masked=torch.full_like(scores,-float('inf'));masked[:,allowed]=scores[:,allowed]
            return masked
    options=dict(do_sample=bool(sampling),num_beams=1,
                 min_new_tokens=0,repetition_penalty=1.,forced_bos_token_id=None,forced_eos_token_id=None)
    if sampling:
        torch.manual_seed(sampling['seed']);options.update({k:sampling[k] for k in ('temperature','top_p','top_k')})
    with torch.inference_mode():
        output=runtime.model.generate(**inputs,max_new_tokens=1,logits_processor=[AnswerOnly()],
            return_dict_in_generate=True,output_logits=True,output_scores=True,
            pad_token_id=runtime.tokenizer.eos_token_id,**options)
        logits=output.logits[0][0].float()
        probabilities=torch.log_softmax(logits,dim=-1)[ids].tolist()
        actual_probabilities=torch.softmax(output.scores[0][0].float(),dim=-1)[ids].tolist()
    selected=int(output.sequences[0,-1]);answer=('Yes','No')[ids.index(selected)]
    if not all(math.isfinite(v) for v in probabilities):raise ValueError('Nonfinite answer-token log probabilities')
    margin=probabilities[0]-probabilities[1]
    measurement=dict(method=VERSION,stage='public_answer_token_before_explanation',margin=margin,stance=math.tanh(margin/2),
        likelihoods=probabilities,token_log_probs=[[v] for v in probabilities],token_counts=[1,1],
        candidate_tokens=[[str(runtime.tokenizer.convert_ids_to_tokens(v))] for v in ids],candidate_token_ids=ids,
        normalized_margin=margin,rendered_prompt=rendered,probe=None,
        selected_answer=answer,selected_token_id=selected,forced=bool(assigned),
        selection_probabilities=actual_probabilities,raw_logits=[float(logits[v]) for v in ids],
        score_preferred_answer='Yes' if margin>0 else 'No' if margin<0 else 'Tie',
        selection_method='forced_candidate' if assigned else 'sampled_constrained_candidates' if sampling else 'candidate_argmax',
        raw_scores='float32 log-softmax before masking/temperature/top-p/top-k; selection probabilities recorded separately')
    del output,logits
    continuation=runtime.encode_rendered(rendered+answer+'\n')['input_ids']
    if continuation[:len(prefix)+1]!=prefix+[selected]:raise ValueError('Explanation prefix changes the selected answer token boundary')
    explanation_sampling=dict(sampling,seed=sampling['seed']+1000003) if sampling else None
    explanation=runtime.generate(messages,max_tokens-1,continuation_prefix=answer+'\n',sampling=explanation_sampling,allow_empty=True)
    argument=explanation['output'].strip()
    generation=dict(output=json.dumps(dict(answer=answer,explanation=argument),ensure_ascii=False),
        complete_output=json.dumps(dict(answer=answer,explanation=argument),ensure_ascii=False),
        generated_tokens=1+explanation['generated_tokens'],max_new_tokens=max_tokens,
        choice_parser_version=PARSER,serialization='application JSON; answer constrained, explanation generated as plain text',
        sampling=sampling,seed=sampling['seed'] if sampling else None,
        answer_token=answer,answer_seed=sampling['seed'] if sampling else None,
        explanation_generation=explanation,raw_explanation=argument,
        explanation_status='valid' if argument else 'empty_explanation',
        hit_token_limit=explanation['generated_tokens']>=max_tokens-1,
        explanation_truncated=explanation.get('finish_reason')=='length',rendered_prompt=rendered,
        continuation_prefix='',**options)
    return dict(answer=answer,argument=argument,measurement=measurement,generation=generation)
