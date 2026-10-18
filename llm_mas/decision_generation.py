"""Capture raw candidate logits at the constrained public-answer generation step."""
import json
import math
import re

VERSION='answer-token-logits-v1'
PARSER='application-serialized-answer-v2'
REASONING_VERSION='answer-token-logits-after-reasoning-v1'
REASONING_PARSER='reasoning-then-answer-v1'
# Protocols whose primary stance is the raw Yes/No answer-token score.
DECISION_PROTOCOLS=('decision_time_v2','reasoning_answer_v3')
# Explicit greedy-compatible settings; checkpoint generation_config defaults
# (for example Qwen2.5's repetition_penalty=1.1) are not inherited.
EXPLICIT_DECODING=dict(num_beams=1,repetition_penalty=1.,no_repeat_ngram_size=0)
ANSWER_MARKER=re.compile(r'(?im)(?:^|(?<=[.!?]))[ \t>*_#-]*(?:final\s+)?answer\s*[*_]*\s*:')
# Structural "Next: continue/finish" lines are never part of the public reasoning text.
NEXT_LINE=re.compile(r'(?im)^[ \t>*_#-]*next\s*[*_]*\s*:.*$')


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
    answer,options,measurement=select_answer(runtime,inputs,ids,sampling,assigned)
    measurement=dict(method=VERSION,stage='public_answer_token_before_explanation',**measurement,rendered_prompt=rendered)
    selected=measurement['selected_token_id']
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


def select_answer(runtime,inputs,ids,sampling=None,assigned=None,labels=('Yes','No')):
    """Select one of two constrained candidate tokens and record its raw same-step scores."""
    torch=runtime.torch
    if assigned not in (None,*labels):raise ValueError('Unknown forced answer')
    if runtime.tokenizer.eos_token_id in ids:raise ValueError('Answer token cannot be EOS')
    allowed=[ids[labels.index(assigned)]] if assigned else ids
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
    selected=int(output.sequences[0,-1]);answer=labels[ids.index(selected)]
    if not all(math.isfinite(v) for v in probabilities):raise ValueError('Nonfinite answer-token log probabilities')
    margin=probabilities[0]-probabilities[1]
    measurement=dict(margin=margin,stance=math.tanh(margin/2),
        likelihoods=probabilities,token_log_probs=[[v] for v in probabilities],token_counts=[1,1],
        candidate_tokens=[[str(runtime.tokenizer.convert_ids_to_tokens(v))] for v in ids],candidate_token_ids=ids,
        normalized_margin=margin,probe=None,
        selected_answer=answer,selected_token_id=selected,forced=bool(assigned),
        selection_probabilities=actual_probabilities,raw_logits=[float(logits[v]) for v in ids],
        score_preferred_answer=labels[0] if margin>0 else labels[1] if margin<0 else 'Tie',
        selection_method='forced_candidate' if assigned else 'sampled_constrained_candidates' if sampling else 'candidate_argmax',
        raw_scores='float32 log-softmax before masking/temperature/top-p/top-k; selection probabilities recorded separately')
    del output,logits
    return answer,options,measurement


def split_reasoning(text):
    """Public reasoning precedes the first answer marker; a leading answer line is dropped."""
    matches=list(ANSWER_MARKER.finditer(text))
    first=matches[0] if matches else None
    start,answer_first=0,False
    if matches and not text[:matches[0].start()].strip():
        answer_first=True
        end=text.find('\n',matches[0].end())
        start=len(text) if end<0 else end+1
        matches=[m for m in matches if m.start()>=start]
    marker=matches[0] if matches else None
    raw=text[start:marker.start() if marker else len(text)].strip()
    reasoning=NEXT_LINE.sub('',raw).strip()
    stated=re.match(r'[\s*_]*(yes|no)\b',text[first.end():],re.I) if first else None
    return dict(reasoning=reasoning,raw_reasoning=raw,marker_found=marker is not None,answer_before_reasoning=answer_first,
                next_before_answer=reasoning!=raw,stated_answer=stated.group(1).capitalize() if stated else None)


def answer_marker_stop(tokenizer,prompt_length,torch):
    from transformers import StoppingCriteria
    class AnswerMarkerStop(StoppingCriteria):
        # Greedy prefixes are unchanged by stopping, so this only saves tokens.
        def __call__(self,input_ids,scores,**kwargs):
            text=tokenizer.decode(input_ids[0,prompt_length:],skip_special_tokens=True)
            parsed=split_reasoning(text)
            done=parsed['marker_found'] and bool(parsed['raw_reasoning'])
            return torch.full((input_ids.shape[0],),done,dtype=torch.bool,device=input_ids.device)
    return AnswerMarkerStop()


def generate_reasoning_first(runtime,messages,max_tokens,own_messages,exempt,copy_block_ngram,sampling=None,assigned=None,next_choice=False):
    """Generate public reasoning, then score the Yes/No token after an appended answer marker."""
    from .copy_block import OwnMessageNgramBlock
    if max_tokens<2:raise ValueError('Reasoning-first responses require at least two tokens')
    torch=runtime.torch
    rendered,inputs=runtime.inputs(messages)
    runtime.check_context(inputs.input_ids.shape[1],reserve_tokens=max_tokens)
    prompt_length=inputs.input_ids.shape[1]
    encode=lambda text:runtime.tokenizer(text,add_special_tokens=False)['input_ids']
    block=OwnMessageNgramBlock([encode(m) for m in own_messages],[encode(e) for e in exempt],copy_block_ngram,prompt_length) if copy_block_ngram and own_messages else None
    reasoning_sampling=dict(sampling) if sampling else None
    raw=runtime.generate(messages,max_tokens-1,sampling=reasoning_sampling,allow_empty=True,decoding=EXPLICIT_DECODING,
                         logits_processor=[block] if block else None,stopping_criteria=[answer_marker_stop(runtime.tokenizer,prompt_length,torch)])
    parsed=split_reasoning(raw['output'])
    reasoning=parsed['reasoning']
    prefix=(reasoning+'\n' if reasoning else '')+'Answer:'
    rendered,inputs=runtime.inputs(messages,prefix)
    runtime.check_context(inputs.input_ids.shape[1],reserve_tokens=1)
    boundary=runtime.encode_rendered(rendered+prefix)['input_ids']
    ids=[]
    for candidate in (' Yes',' No'):
        full=runtime.encode_rendered(rendered+prefix+candidate)['input_ids']
        if full[:len(boundary)]!=boundary or len(full)!=len(boundary)+1:
            raise ValueError('Reasoning-first Yes/No candidates must each be one token after "Answer:"; checkpoint unsupported')
        ids.append(full[-1])
    if len(set(ids))!=2:raise ValueError('Yes and No must have distinct token IDs')
    answer_sampling=dict(sampling,seed=sampling['seed']+1000003) if sampling else None
    answer,options,measurement=select_answer(runtime,inputs,ids,answer_sampling,assigned)
    measurement=dict(method=REASONING_VERSION,stage='answer_token_after_reasoning',**measurement,rendered_prompt=rendered+prefix)
    message=prefix+' '+answer
    next_action,next_measurement=(select_next(runtime,messages,message,sampling) if next_choice else (None,None))
    if next_action:
        message+='\nNext: '+next_action
    copy_block=dict(ngram=copy_block_ngram,source='own_previous_public_messages',exempt='system_question_and_turn_instruction',
                    banned_ngrams=block.banned_ngrams if block else 0,blocked_steps=block.blocked_steps if block else 0)
    generation=dict(output=raw['output'],complete_output=json.dumps(dict(answer=answer,explanation=reasoning),ensure_ascii=False),
        generated_tokens=raw['generated_tokens']+1,max_new_tokens=max_tokens,
        choice_parser_version=REASONING_PARSER,serialization='reasoning generated first; answer token scored after an appended "Answer:" marker',
        sampling=sampling,seed=sampling['seed'] if sampling else None,
        answer_token=answer,answer_seed=answer_sampling['seed'] if answer_sampling else None,
        reasoning_generation=raw,raw_reasoning_output=raw['output'],reasoning=reasoning,
        reasoning_status='valid' if reasoning else 'empty_reasoning',
        answer_marker_found=parsed['marker_found'],answer_before_reasoning=parsed['answer_before_reasoning'],
        next_before_answer=parsed['next_before_answer'],
        stated_answer=parsed['stated_answer'],stated_answer_matches=parsed['stated_answer']==answer if parsed['stated_answer'] else None,
        decoding=dict(EXPLICIT_DECODING,do_sample=bool(sampling)),copy_block=copy_block,
        hit_token_limit=raw['generated_tokens']>=max_tokens-1,
        reasoning_truncated=raw.get('finish_reason')=='length',rendered_prompt=raw['rendered_prompt'],
        continuation_prefix='',answer_options=options,next_action=next_action)
    return dict(answer=answer,argument=reasoning,message=message,measurement=measurement,generation=generation,
                next_action=next_action,next_measurement=next_measurement)


def select_next(runtime,messages,message,sampling=None):
    """Score the conversation signal after the answer line: " finish" vs " continue".

    stance > 0 favors finish. Single-token candidates use the same constrained-token step as the
    answer; otherwise whole-word likelihoods are compared (argmax, also under sampling).
    """
    prefix=message+'\nNext:'
    labels=('finish','continue')
    rendered,inputs=runtime.inputs(messages,prefix)
    runtime.check_context(inputs.input_ids.shape[1],reserve_tokens=1)
    boundary=runtime.encode_rendered(rendered+prefix)['input_ids']
    ids=[]
    for label in labels:
        full=runtime.encode_rendered(rendered+prefix+' '+label)['input_ids']
        if full[:len(boundary)]!=boundary or len(full)!=len(boundary)+1:break
        ids.append(full[-1])
    if len(ids)==2 and ids[0]!=ids[1]:
        next_sampling=dict(sampling,seed=sampling['seed']+2000003) if sampling else None
        action,_,measurement=select_answer(runtime,inputs,ids,next_sampling,labels=labels)
        measurement['method']='next-token-logits-v1'
    else:
        scores=[runtime.likelihood(messages,' '+label,continuation_prefix=prefix)[0] for label in labels]
        margin=scores[0]-scores[1]
        action=labels[0] if margin>=0 else labels[1]
        measurement=dict(method='next-word-likelihood-v1',margin=margin,likelihoods=scores,selection_method='candidate_argmax')
    measurement=dict(measurement,stance=math.tanh(measurement['margin']/2),candidates=list(labels),selected_action=action,
                     stance_meaning='positive favors finish',rendered_prompt=rendered+prefix)
    measurement.pop('selected_answer',None)
    if 'score_preferred_answer' in measurement:measurement['score_preferred_action']=measurement.pop('score_preferred_answer')
    return action,measurement
