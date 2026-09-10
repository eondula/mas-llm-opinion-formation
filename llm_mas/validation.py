"""Candidate checks on exact recorded histories and explicit-choice elicitation."""
from copy import deepcopy
import re


def annotate_choice(text):
    # Only accept a leading explicit option; ambiguous responses remain unparsed.
    match=re.match(r'^\s*(?:I\s+(?:choose|select|prefer)\s+)?(?:Option\s+)?([AB])(?:\b|[.:])',text,re.I)
    return match.group(1).upper() if match else None


def evaluate(runtime, history, scenario, texts, max_tokens=128):
    baseline=runtime.measure(history,scenario,texts)
    variants={}
    pairs=scenario.get('candidate_variants',[])
    for i,pair in enumerate(pairs):
        s=deepcopy(scenario)
        for k,value in zip(('strategy_1','strategy_2'),pair):s[k]['candidate']=value
        variants[f'paraphrase_{i}']=runtime.measure(history,s,texts)
    swapped=deepcopy(scenario)
    swapped['strategy_1'],swapped['strategy_2']=swapped['strategy_2'],swapped['strategy_1']
    reverse=runtime.measure(history,swapped,texts)
    independent=[]
    for order in ((1,2),(2,1)):
        instruction='Choose between these alternatives. Begin with exactly A or B, then briefly explain.\n'+ '\n'.join(f'{label}: {scenario[f"strategy_{strategy}"]["candidate"]}' for label,strategy in zip(('A','B'),order))
        generated=runtime.generate(history+[{'role':'user','content':instruction}],max_tokens)
        choice=annotate_choice(generated['output'])
        independent.append(dict(order=order,**generated,parsed_choice=choice,semantic_strategy=order[0 if choice=='A' else 1] if choice else None))
    lengths=[len(x) for x in baseline.get('token_log_probs',[[],[]])]
    likelihoods=baseline.get('likelihoods',[None,None])
    normalized=(likelihoods[0]/lengths[0]-likelihoods[1]/lengths[1]) if all(lengths) else None
    return dict(baseline=baseline,repeat=runtime.measure(history,scenario,texts),swapped_candidates=reverse,
                swapped_semantic_margin=-reverse['margin'],variants=variants,independent=independent,
                normalized_margin=normalized,history=history,
                paraphrase_status='evaluated' if pairs else 'not_supplied')
