#!/usr/bin/env python3
"""Question-centered descriptive analysis of saved deliberation; no model inference."""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import statistics

from model_saved_pilot import export, hashes

COMPOSITIONS=('QQ','SS','QS','SQ')
STATES=('Yes','No','Split','Unknown')
REFERENCE=re.compile(r'\bagent\s+[ab]\b|\b(?:your|the other agent.s|my peer.s)\s+(?:argument|point|claim|reasoning|response)\b|\b(?:as you (?:said|noted|mentioned)|agree with you|disagree with you)\b',re.I)
AGREEMENT=re.compile(r'\b(?:i agree|you are right|you.re right|i accept|agree with)\b',re.I)
CHALLENGE=re.compile(r'\b(?:i disagree|you are wrong|your (?:argument|claim|reasoning) is (?:wrong|incorrect|flawed)|disagree with)\b',re.I)


def normalized(text):
    return ' '.join(re.findall(r'\w+',text.casefold()))


def tokens(text):
    return set(normalized(text).split())


def jaccard(left,right):
    a,b=tokens(left),tokens(right)
    return len(a&b)/len(a|b) if a|b else None


def valid(row):
    return row.get('choice_status')=='valid' and row.get('answer') in ('Yes','No')


def state(rows):
    if not all(valid(r) for r in rows):return 'Unknown'
    return rows[0]['answer'] if len({r['answer'] for r in rows})==1 else 'Split'


def mean(values):
    selected=[v for v in values if v is not None]
    return statistics.mean(selected) if selected else None


def link(conversation,response):
    return f'{conversation}__{response}'


def load(source):
    records=[]
    for file in sorted((source/'conversations').glob('*.json')):
        record=json.loads(file.read_text());record['_conversation']=file.stem
        models=record['config']['models']
        if len(models)!=2 or any('Qwen' not in m and 'SmolLM2' not in m for m in models):raise ValueError('This report expects the reviewed two-checkpoint pairwise pilot')
        record['_composition']=''.join('Q' if 'Qwen' in m else 'S' for m in models)
        if record['config']['schedule'] not in ('synchronous','no_peer'):raise ValueError('This report expects synchronous matched controls')
        records.append(record)
    if not records:raise ValueError('No saved conversations')
    if len({str(r['config'].get('initialization')) for r in records})!=1:raise ValueError('Do not pool natural and forced initialization')
    return records


def analyze(records):
    turns=[];trajectories=[];groups=[];conversations=[];deliveries=[];pairs=defaultdict(list)
    for d in records:
        conv=d['_conversation'];comp=d['_composition'];schedule=d['config']['schedule'];rows=d['rows'];label=d['evaluation'].get('reference_answer')
        base=dict(question_id=d['question_id'],composition=comp,schedule=schedule,conversation=conv,repetition=d.get('repetition'),pair_id=d.get('pair_id'))
        byid={r['response_id']:(i,r) for i,r in enumerate(rows)};byagent=defaultdict(list);byround=defaultdict(list);previous={}
        for i,r in enumerate(rows):
            prior=previous.get(r['agent']);previous[r['agent']]=r;byagent[r['agent']].append(r);byround[r['round']].append(r)
            rowbase=dict(base,response_id=r['response_id'],agent=r['agent'],model=r['model'],round=r['round'])
            incoming=[]
            for delivery in r.get('deliveries',[]):
                index,peer=byid[delivery['source_id']]
                if index>=i or peer['round']!=r['round']-1 or peer['message']!=delivery['message']:raise ValueError('Invalid delivery timing/content')
                if schedule=='no_peer':raise ValueError('Unexpected control delivery')
                incoming.append(peer)
                argument=r.get('argument','');peerarg=peer.get('argument','');oldarg=prior.get('argument','') if prior else ''
                ref=REFERENCE.search(argument);agree=AGREEMENT.search(argument);challenge=CHALLENGE.search(argument)
                answer_eligible=valid(r) and prior is not None and valid(prior) and valid(peer)
                deliveries.append(dict(rowbase,source_id=peer['response_id'],source_round=peer['round'],speaker=peer['agent'],
                    prior_answer=prior['answer'] if prior and valid(prior) else 'Invalid',peer_answer=peer['answer'] if valid(peer) else 'Invalid',answer=r['answer'] if valid(r) else 'Invalid',
                    answer_eligible=answer_eligible,reference_cue=bool(ref),reference_cue_text=ref.group(0) if ref else '',
                    agreement_cue=bool(agree),challenge_cue=bool(challenge),
                    exact_peer_argument_reuse=bool(normalized(argument)) and normalized(argument)==normalized(peerarg),
                    exact_self_argument_repeat=bool(normalized(argument)) and normalized(argument)==normalized(oldarg),
                    peer_token_jaccard=jaccard(argument,peerarg),self_token_jaccard=jaccard(argument,oldarg),
                    changed_to_peer_answer=(r['answer']==peer['answer'] and prior['answer']!=peer['answer']) if answer_eligible else None,
                    retained_answer_against_peer=(r['answer']==prior['answer'] and r['answer']!=peer['answer']) if answer_eligible else None,
                    movement_toward_peer=(abs(r['post']['stance']-peer['post']['stance'])<abs(prior['post']['stance']-peer['post']['stance'])-.01) if prior else None,
                    prior_message=prior['message'] if prior else '',delivered_message=delivery['message'],response_message=r['message'],
                    interpretation_status='automated text features; semantic interpretation unreviewed',
                    transcript=f'questions/{d["question_id"]}.html#{link(conv,r["response_id"])}'))
            transition_valid=prior is not None and valid(prior) and valid(r)
            adjacent=r.get('message_similarity',{}).get('adjacent') if prior and transition_valid else None
            turns.append(dict(rowbase,answer=r['answer'] if valid(r) else 'Invalid',valid=valid(r),reference_label=label,
                correct=(r['answer']==label and valid(r)) if label in ('Yes','No') else None,
                margin=r['post']['margin'],stance=r['post']['stance'],sentiment=r.get('sentiment'),
                prior_answer=prior['answer'] if prior and valid(prior) else ('Invalid' if prior else None),
                eligible_transition=transition_valid,answer_reversal=(r['answer']!=prior['answer']) if transition_valid else None,
                correct_to_incorrect=(prior['answer']==label and r['answer']!=label) if transition_valid and label in ('Yes','No') else None,
                incorrect_to_correct=(prior['answer']!=label and r['answer']==label) if transition_valid and label in ('Yes','No') else None,
                stance_change=r['post']['stance']-prior['post']['stance'] if prior else None,
                adjacent_cosine=adjacent,exact_argument_repeat=bool(normalized(r.get('argument',''))) and normalized(r.get('argument',''))==normalized(prior.get('argument','')) if prior else None,
                wording_changed=normalized(r.get('argument',''))!=normalized(prior.get('argument','')) if prior else None,
                message=r['message'],argument=r.get('argument',''),delivery_count=len(incoming),
                transcript=f'questions/{d["question_id"]}.html#{link(conv,r["response_id"])}'))
        groupsequence=[]
        for turn,rr in sorted(byround.items()):
            if len(rr)!=2:raise ValueError('Incomplete recorded round')
            groupstate=state(rr);groupsequence.append(groupstate)
            groups.append(dict(base,round=turn,state=groupstate,valid_group=groupstate!='Unknown',
                consensus=groupstate in ('Yes','No'),correct_consensus=(groupstate==label) if label in ('Yes','No') else None,
                stance_dispersion=statistics.pstdev(r['post']['stance'] for r in rr),
                yes_answers=sum(valid(r) and r['answer']=='Yes' for r in rr),no_answers=sum(valid(r) and r['answer']=='No' for r in rr),invalid_answers=sum(not valid(r) for r in rr)))
        for agent,rr in byagent.items():
            rr=sorted(rr,key=lambda r:r['round']);own=[t for t in turns if t['conversation']==conv and t['agent']==agent]
            reversals=sum(t['answer_reversal'] is True for t in own);fullyvalid=all(valid(r) for r in rr)
            aba=sum(all(valid(v) for v in rr[i-2:i+1]) and rr[i]['answer']==rr[i-2]['answer']!=rr[i-1]['answer'] for i in range(2,len(rr)))
            trajectories.append(dict(base,reference_label=label,agent=agent,model=rr[0]['model'],initial_answer=rr[0]['answer'] if valid(rr[0]) else 'Invalid',final_answer=rr[-1]['answer'] if valid(rr[-1]) else 'Invalid',
                initial_margin=rr[0]['post']['margin'],final_margin=rr[-1]['post']['margin'],initial_stance=rr[0]['post']['stance'],final_stance=rr[-1]['post']['stance'],
                initial_correct=valid(rr[0]) and rr[0]['answer']==label,final_correct=valid(rr[-1]) and rr[-1]['answer']==label,
                fully_valid=fullyvalid,invalid_responses=sum(not valid(r) for r in rr),reversals=reversals,eligible_transitions=sum(t['eligible_transition'] for t in own),
                persistent_answer=fullyvalid and reversals==0,repeated_switching=reversals>=2,aba_oscillations=aba,
                absolute_stance_change=mean([abs(t['stance_change']) for t in own if t['stance_change'] is not None]),
                argument_repeat_rate=mean([t['exact_argument_repeat'] for t in own if t['round']>0]),adjacent_cosine=mean([t['adjacent_cosine'] for t in own])))
        first_consensus=next((i for i,s in enumerate(groupsequence) if s in ('Yes','No')),None)
        durable=next((i for i,s in enumerate(groupsequence) if s in ('Yes','No') and all(v==s for v in groupsequence[i:])),None)
        conversations.append(dict(base,initial_state=groupsequence[0],final_state=groupsequence[-1],
            first_consensus_round=first_consensus,durable_consensus_round=durable,
            final_consensus=groupsequence[-1] in ('Yes','No'),initial_disagreement=groupsequence[0]=='Split',
            maintained_disagreement=all(s=='Split' for s in groupsequence),
            initial_disagreement_to_consensus=groupsequence[0]=='Split' and groupsequence[-1] in ('Yes','No'),
            initial_consensus_lost=groupsequence[0] in ('Yes','No') and any(s!=groupsequence[0] for s in groupsequence[1:]),
            consensus_then_disagreement=any(groupsequence[i] in ('Yes','No') and 'Split' in groupsequence[i+1:] for i in range(len(groupsequence))),
            invalid_rounds=sum(s=='Unknown' for s in groupsequence),
            mean_dispersion=mean([g['stance_dispersion'] for g in groups if g['conversation']==conv])))
        pairs[d.get('pair_id')].append(d)
    pairing=[]
    for pairid,dd in sorted(pairs.items()):
        if pairid is None or len(dd)!=2 or {d['config']['schedule'] for d in dd}!={'synchronous','no_peer'}:raise ValueError('Unmatched pair')
        syn=next(d for d in dd if d['config']['schedule']=='synchronous');ctl=next(d for d in dd if d['config']['schedule']=='no_peer')
        left=[r for r in syn['rows'] if r['round']==0];right=[r for r in ctl['rows'] if r['round']==0]
        if left!=right or syn.get('initialization_hash')!=ctl.get('initialization_hash'):raise ValueError('Initializations are not exactly matched')
        own=[t for t in trajectories if t['pair_id']==pairid]
        def metric(schedule,key):return mean([t[key] for t in own if t['schedule']==schedule])
        states={s:next(c for c in conversations if c['pair_id']==pairid and c['schedule']==s) for s in ('synchronous','no_peer')}
        row=dict(pair_id=pairid,question_id=syn['question_id'],composition=syn['_composition'],repetition=syn.get('repetition'),initialization_exact_match=True,
                 discussion_conversation=syn['_conversation'],control_conversation=ctl['_conversation'])
        for key in ('initial_correct','final_correct','reversals','absolute_stance_change','argument_repeat_rate'):
            row[key+'_discussion']=metric('synchronous',key);row[key+'_control']=metric('no_peer',key);row[key+'_difference']=row[key+'_discussion']-row[key+'_control']
        row.update(discussion_final_state=states['synchronous']['final_state'],control_final_state=states['no_peer']['final_state'],consensus_difference=int(states['synchronous']['final_consensus'])-int(states['no_peer']['final_consensus']))
        pairing.append(row)
    return turns,trajectories,groups,conversations,deliveries,pairing


def summaries(turns,trajectories,groups,conversations,deliveries,keys):
    buckets=defaultdict(list)
    for r in trajectories:buckets[tuple(r[k] for k in keys)].append(r)
    output=[]
    for values,rr in sorted(buckets.items()):
        ids={r['conversation'] for r in rr};tt=[r for r in turns if r['conversation'] in ids];gg=[r for r in groups if r['conversation'] in ids];cc=[r for r in conversations if r['conversation'] in ids];dd=[r for r in deliveries if r['conversation'] in ids]
        updates=[r for r in tt if r['round']>0]
        row=dict(zip(keys,values),conversations=len(cc),agent_trajectories=len(rr),questions=len({r['question_id'] for r in rr}),recorded_responses=len(tt),
            initial_accuracy=mean([r['initial_correct'] for r in rr if r['initial_answer'] in ('Yes','No')]),final_accuracy=mean([r['final_correct'] for r in rr if r['final_answer'] in ('Yes','No')]),
            initial_valid_answers=sum(r['initial_answer'] in ('Yes','No') for r in rr),final_valid_answers=sum(r['final_answer'] in ('Yes','No') for r in rr),
            initial_invalid_answers=sum(r['initial_answer']=='Invalid' for r in rr),final_invalid_answers=sum(r['final_answer']=='Invalid' for r in rr),
            initial_disagreement_rate=mean([r['initial_disagreement'] for r in cc]),initial_invalid_groups=sum(r['initial_state']=='Unknown' for r in cc),
            final_consensus_rate=mean([r['final_consensus'] for r in cc]),final_unknown_groups=sum(r['final_state']=='Unknown' for r in cc),
            maintained_disagreement_rate=mean([r['maintained_disagreement'] for r in cc]),
            disagreement_to_consensus_rate=mean([r['initial_disagreement_to_consensus'] for r in cc if r['initial_disagreement']]),
            persistent_answer_rate=mean([r['persistent_answer'] for r in rr if r['fully_valid']]),persistent_eligible=sum(r['fully_valid'] for r in rr),
            repeated_switching_rate=mean([r['repeated_switching'] for r in rr]),total_reversals=sum(r['reversals'] for r in rr),
            reversal_rate=mean([r['answer_reversal'] for r in updates if r['eligible_transition']]),eligible_transitions=sum(r['eligible_transition'] for r in updates),
            aba_oscillations=sum(r['aba_oscillations'] for r in rr),invalid_response_rate=mean([not r['valid'] for r in tt]),
            mean_absolute_stance_update=mean([abs(r['stance_change']) for r in updates]),
            exact_self_repeat_rate=mean([r['exact_argument_repeat'] for r in updates]),adjacent_cosine=mean([r['adjacent_cosine'] for r in updates]),
            reference_cue_rate=mean([r['reference_cue'] for r in dd]),agreement_cue_rate=mean([r['agreement_cue'] for r in dd]),challenge_cue_rate=mean([r['challenge_cue'] for r in dd]),
            exact_peer_reuse_rate=mean([r['exact_peer_argument_reuse'] for r in dd]),delivered_events=len(dd),
            changed_to_peer_answer_rate=mean([r['changed_to_peer_answer'] for r in dd]),
            correct_to_incorrect=sum(r['correct_to_incorrect'] is True for r in updates),incorrect_to_correct=sum(r['incorrect_to_correct'] is True for r in updates))
        output.append(row)
    return output


def endpoint_status(row,endpoint):
    if row[endpoint+'_answer'] not in ('Yes','No'):return 'Invalid'
    if row.get('reference_label') not in ('Yes','No'):return 'Unlabeled'
    return 'Correct' if row[endpoint+'_correct'] else 'Incorrect'


def endpoint_transitions(trajectories,keys):
    buckets=defaultdict(list)
    for r in trajectories:buckets[tuple(r[k] for k in keys)].append(r)
    result=[]
    for values,rows in sorted(buckets.items()):
        statuses=('Correct','Incorrect','Invalid')
        if any(endpoint_status(r,e)=='Unlabeled' for r in rows for e in ('initial','final')):statuses+=('Unlabeled',)
        counts=Counter((endpoint_status(r,'initial'),endpoint_status(r,'final')) for r in rows)
        for initial in statuses:
            for final in statuses:
                n=counts[(initial,final)]
                result.append(dict(zip(keys,values),initial_outcome=initial,final_outcome=final,count=n,total_trajectories=len(rows),proportion=n/len(rows)))
    return result


def coding_criteria():
    return [dict(code='reference cue',definition='Regex for explicit Agent A/B naming, your argument/point/claim/reasoning/response, or selected direct peer-address phrases.',status='Automatic surface feature; may reference self or misattribute speaker; not confirmed engagement.'),
        dict(code='agreement/challenge cue',definition='Narrow explicit phrases such as I agree / I disagree / you are right / your claim is wrong.',status='Automatic surface feature; absence does not imply no agreement or challenge. However alone is not a challenge code.'),
        dict(code='exact argument repetition/reuse',definition='Nonempty explanations equal after case-folding and word-token whitespace normalization; compares own prior or delivered peer explanation.',status='Observable textual equivalence; does not establish causal copying or semantic equivalence of different wording.'),
        dict(code='wording changed',definition='Normalized explanation differs from own immediately previous explanation.',status='Textual change only; cannot establish argument revision, novelty, or improved reasoning.'),
        dict(code='answer alignment',definition='Valid current, previous, delivered-peer answers; previous differs from peer and current equals peer.',status='Categorical association; not proof of argument adoption.'),
        dict(code='reference (interpretive)',definition='Response explicitly points to the delivered speaker or a claim in the delivered message; compare exact text and attribution.',status='Requires excerpt inspection; annotation examples are assistant-reviewed, not independent human coding.'),
        dict(code='challenge (interpretive)',definition='Response rejects, qualifies or counters a substantive proposition in the delivered message; record the proposition and response.',status='Requires contextual inspection; different Yes/No answers alone are insufficient.'),
        dict(code='adoption/reuse (interpretive)',definition='Response incorporates a claim or reasoning from the delivered message; compare prior own response to distinguish pre-existing content.',status='Requires contextual inspection; shared wording is evidence of reuse, not causal proof.'),
        dict(code='no observable uptake / ignores',definition='Record unchanged own wording despite a conflicting delivered answer as no observable textual uptake. Do not label psychological ignoring from absence of reference.',status='Ignoring cannot generally be established from transcripts. Leave semantic ignore code indeterminate.'),
        dict(code='answer persistence / reversal / repeated switching',definition='Persistence requires all six answers valid and unchanged. Reversals require adjacent valid answers; invalid rounds are never bridged. Repeated switching means at least two observed reversals.',status='Repeated switching may be a lower bound when invalid responses occur; eligibility is exported.'),
        dict(code='ABA oscillation',definition='Three consecutive valid answers A,B,A with A different from B.',status='Exact categorical definition, distinct from score fluctuation.'),
        dict(code='consensus / maintained disagreement',definition='Consensus is two valid identical answers. Maintained disagreement requires Split in every recorded round. Unknown marks any invalid answer.',status='First consensus can be transient; durable consensus is the earliest round with one unanimous answer unchanged through the recorded end, not a guarantee beyond it.')]


def annotated_examples(records,deliveries):
    specifications=[
        ('006_004','r1_a0','explicit reference and claim reuse','The response names Agent B and repeats the chess-experience claim in the delivered text. It treats the agent as the subject of the experience claim, illustrating attribution confusion. This is not evidence of correct reasoning.'),
        ('006_004','r2_a0','substantial delivered-claim reuse','The response reproduces most of the delivered explanation while changing the opening subject to Agent B. Inspect the prior own message to see the expansion in claims.'),
        ('006_007','r1_a1','reference with attribution/content mismatch','The response says Agent B is correct and asserts a discount, but the actually delivered message argues No. Agent B is also the responding agent. Explicit naming is therefore not reliable evidence of faithful peer engagement.'),
        ('007_005','r1_a1','shared factual phrase, opposite categorical answer','The response reuses the delivered -123°C temperature but concludes No where the delivered message concludes Yes. Shared factual wording does not imply answer agreement.'),
        ('000_003','r5_a0','qualified counterclaim','The delivered message denies an ability to develop spider veins even under extreme conditions. The response retains No but asserts analogous structural features under environmental stress. This is a substantive qualification without a direct I disagree cue; factual validity is not adjudicated.'),
        ('001_003','r4_a1','near-verbatim delivered explanation reuse','The response closely reproduces the delivered explanation with a small wording change. Prior-own text is retained for comparison; reuse does not establish causal influence.'),
        ('001_003','r5_a0','exact delivered explanation reuse','The public response repeats the delivered explanation. This illustrates textual convergence; it does not establish independent corroboration or correctness.'),
        ('000_008','r2_a0','answer reversal with related temporal claim','The agent changes Yes to No after a delivered No argument about vehicle availability. Its response likewise argues from historical availability. This is an observable alignment and related claim, not a causal attribution or fact check.'),
        ('001_002','r1_a1','answer alignment with argument reframing','The agent changes No to Yes, replacing an argument about unrelated identities with a discussion of possible celebrity connections similar to the delivered framing. This illustrates a contextual reasoning shift rather than mere answer repetition.'),
        ('000_006','r2_a0','revised timing claim with answer reversal','After the delivered Yes argument gives an approximately 11:45 PM starting time and earlier moonset, the next response changes No to Yes and uses a similar timing account. Neither the stated schedule nor moonset is independently verified here.'),
        ('001_001','r1_a0','aligned answer with indirect-observation qualification','The response changes No to Yes after a delivered Yes message, while retaining the limitation on direct imaging and introducing indirect sensing. Agreement in answer coexists with qualification of what observation means.'),
        ('000_000','r1_a0','explicit uptake of a delivered ethical-counterexample account','The response changes Yes to No and explicitly refers to the mention of instances where ethical concerns arose, which appears in the delivered message. This is strong textual evidence of engagement, without establishing factual validity or a causal mechanism.'),
        ('001_009','r3_a1','no observable textual uptake','The prior own explanation is repeated unchanged despite a delivered No explanation. The response remains Yes. This supports no observable textual revision in this turn, not a claim that the agent internally ignored the message.')]
    indexed={(r['conversation'],r['response_id']):r for r in deliveries};output=[]
    for conv,response,code,note in specifications:
        row=indexed.get((conv,response))
        if row:
            output.append(dict(row,reviewed_interpretation=code,interpretation_note=note,review_status='assistant-inspected illustrative case; purposive selection, not a representative coded sample'))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/experiments/new_exp_1_a'))
    parser.add_argument('--modeling',type=Path,default=Path('results/modeling/new_exp_1_a_system_identification'))
    parser.add_argument('--output',type=Path,default=Path('results/deliberation/new_exp_1_a_question_report'))
    args=parser.parse_args();source,modeling,output=(p.resolve() for p in (args.source,args.modeling,args.output))
    if output.exists():parser.error('Choose a fresh output directory')
    for protected in (source,modeling):
        if output==protected or protected in output.parents or output in protected.parents:parser.error('Output must be separate from experiment and modeling inputs')
    source_hashes=hashes(source);model_hashes=hashes(modeling) if modeling.exists() else {}
    records=load(source);turns,trajectories,groups,conversations,deliveries,pairs=analyze(records)
    benchmarks=[];failures=[]
    if model_hashes:
        provenance=json.loads((modeling/'manifest.json').read_text())
        if provenance['source_sha256']!=source_hashes:raise ValueError('Modeling inputs do not match experiment')
        benchmarks=json.loads((modeling/'summary.json').read_text())
        failures=sorted([r for r in json.loads((modeling/'predictions.json').read_text()) if r['subset']=='all_finite' and r['method']=='coupled' and r['mode']=='one_step' and r['schedule']=='synchronous'],key=lambda r:abs(r['prediction']-r['target_stance']),reverse=True)[:20]
        failures=[dict(question_id=r['question'],conversation=r['conversation'],response_id=r['response_id'],composition=''.join('Q' if 'Qwen' in m else 'S' for m in r['composition'].split(' | ')),round=r['round'],predicted_stance=r['prediction'],observed_stance=r['target_stance'],absolute_error=abs(r['prediction']-r['target_stance']),transcript=f'questions/{r["question"]}.html#{link(r["conversation"],r["response_id"])}') for r in failures]
    if hashes(source)!=source_hashes or (model_hashes and hashes(modeling)!=model_hashes):raise RuntimeError('Input changed during analysis')
    output.mkdir(parents=True);tables=output/'tables';tables.mkdir()
    configuration=summaries(turns,trajectories,groups,conversations,deliveries,('composition','schedule'))
    question=summaries(turns,trajectories,groups,conversations,deliveries,('question_id','composition','schedule'))
    examples=annotated_examples(records,deliveries)
    export(tables,'initial_final_transitions',endpoint_transitions(trajectories,('composition','schedule')))
    export(tables,'question_initial_final_transitions',endpoint_transitions(trajectories,('question_id','composition','schedule')))
    for name,data in [('agent_turns',turns),('agent_trajectories',trajectories),('group_rounds',groups),('conversation_patterns',conversations),('delivered_message_features',deliveries),('matched_pairs',pairs),('configuration_summary',configuration),('question_configuration_summary',question),('coding_criteria',coding_criteria()),('illustrative_excerpts',examples),('model_benchmarks',benchmarks),('model_failure_excerpts',failures)]:export(tables,name,data)
    review=[dict(question_id=r['question_id'],conversation=r['conversation'],response_id=r['response_id'],source_id=r['source_id'],transcript=r['transcript'],reference='',challenge='',adoption='',no_observable_uptake='',ignoring='indeterminate',notes='') for r in deliveries]
    export(tables,'semantic_review_template',review)
    manifest=dict(source=str(source),source_sha256=source_hashes,modeling=str(modeling),modeling_sha256=model_hashes,inputs_unchanged=True,
        scripts_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('report_question_deliberation.py','deliberation_report_figures.py','model_saved_pilot.py')},
        conversations=len(records),question_count=len({r['question_id'] for r in turns}),paired_blocks=len(pairs),recorded_responses=len(turns),
        independent_initial_generations=sum(r['round']==0 and r['schedule']=='synchronous' for r in turns),
        actual_generations=sum(r['round']>0 or r['schedule']=='synchronous' for r in turns),
        notes=['Initial/final accuracy is conditional on valid explicit answers; endpoint invalid counts and valid denominators are exported separately. Endpoint transition matrices include invalid outcomes as their own category.',
            'Round-zero records are duplicated across matched branches; do not count them as independent generations.',
            'All composition comparisons retain same questions; pooled rates are descriptive of this balanced pilot. Missing/conditional denominators are exported.',
            'Textual cues are not semantic labels. Wording changes are not necessarily reasoning revisions. No observable uptake is not internal ignoring.',
            'Thirteen assistant-inspected purposive excerpts illustrate patterns; no independent semantic annotation or inter-rater reliability is claimed.',
            'Existing adjacent cosine scores are reused; no embeddings or LLMs are loaded. Similarity is not proof of semantic agreement.',
            'Stance scales are not calibrated across checkpoints. Correctness is only one outcome; labels remain evaluator-only.',
            'Matched controls support protocol comparisons; individual movement does not establish a causal mechanism.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    from deliberation_report_figures import generate
    generate(output,records,turns,trajectories,groups,conversations,deliveries,pairs,configuration,question,examples,benchmarks,failures)
    print(output/'index.html')


if __name__=='__main__':main()
