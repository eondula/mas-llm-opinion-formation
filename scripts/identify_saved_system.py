#!/usr/bin/env python3
"""Configuration-aware identification of saved LLM output transitions; no inference."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from model_saved_pilot import export, hashes, simplex_fit
from evaluate_saved_outcomes import observations, build as outcome_rows, summarize as outcome_summary

METHODS = ('persistence', 'control_drift', 'self_dynamics', 'coupled', 'configuration_coupled', 'degroot', 'friedkin_johnsen')
RIDGE_METHODS = ('self_dynamics', 'coupled', 'configuration_coupled')
ALPHAS = (.01, .1, 1., 10., 100.)


def extract(source):
    result = []
    for file in sorted((source/'conversations').glob('*.json')):
        record = json.loads(file.read_text())
        rows = record['rows']
        lookup = {r['response_id']: (i,r) for i,r in enumerate(rows)}
        initial = {r['agent']: r for r in rows if r['round']==0}
        previous = dict(initial)
        models = record['config']['models']
        if record['config']['schedule'] not in ('synchronous','no_peer','sequential','reverse_sequential'):
            raise ValueError('Unsupported delivery schedule')
        for index, row in enumerate(rows):
            if not row['round']:
                continue
            own = previous[row['agent']]
            first = initial[row['agent']]
            peers = []
            for delivery in row.get('deliveries', []):
                source_index, peer = lookup[delivery['source_id']]
                if source_index >= index or peer['message'] != delivery['message'] or delivery.get('speaker') != peer['agent']:
                    raise ValueError(f'Invalid delivery at {file}:{row["response_id"]}')
                if record['config']['schedule']=='synchronous' and peer['round'] != row['round']-1:
                    raise ValueError('Synchronous delivery must use previous-round output')
                peers.append(peer)
            if record['config']['schedule']=='no_peer' and peers:
                raise ValueError('No-peer condition contains deliveries')
            values = [r['post']['margin'] for r in [own,first,row]+peers]
            if not all(np.isfinite(v) for v in values):
                raise ValueError('Nonfinite score; explicit missing-score handling required')
            if abs(row['before']['margin']-own['post']['margin'])>1e-6:
                raise ValueError('Own-state continuity mismatch')
            slot = list(initial).index(row['agent'])
            result.append(dict(question=record['question_id'],conversation=file.stem,response_id=row['response_id'],
                agent=row['agent'],slot=slot,model=row['model'],composition=' | '.join(models),
                configuration=f'{" | ".join(models)};slot={slot}',schedule=record['config']['schedule'],
                initialization=str(record['config']['initialization']),repetition=record.get('repetition'),pair_id=record.get('pair_id'),
                round=row['round'],prior_id=own['response_id'],peer_ids='|'.join(r['response_id'] for r in peers),
                own=float(own['post']['margin']),initial=float(first['post']['margin']),
                peer=float(np.mean([r['post']['margin'] for r in peers])) if peers else float(own['post']['margin']),
                peer_count=len(peers),target=float(row['post']['margin']),
                valid=all(r.get('choice_status')=='valid' for r in [own,first,row]+peers)))
            previous[row['agent']]=row
    if len({r['question'] for r in result})<4:
        raise ValueError('Nested question validation requires at least four questions')
    if len({r['initialization'] for r in result})!=1:
        raise ValueError('Analyze natural and forced starts separately')
    paired=defaultdict(list)
    for row in result:
        if row['pair_id'] and row['round']==1:
            paired[(row['pair_id'],row['agent'])].append(row)
    for key, starts in paired.items():
        if len(starts)!=2 or {r['schedule'] for r in starts}!={'synchronous','no_peer'} or starts[0]['own']!=starts[1]['own']:
            raise ValueError(f'Unmatched paired initialization: {key}')
    return result


def feature_names(method, configurations):
    names=['intercept','own','initial_minus_own','round','slot','has_peer']
    if method!='self_dynamics':
        names+=['peer_minus_own']
    if method=='configuration_coupled':
        for configuration in configurations:
            names += [f'{configuration}::{n}' for n in ('offset','own','initial_minus_own','round','peer_minus_own')]
    return names


def design(rows, method, configurations):
    matrix=[]
    for r in rows:
        gap=r['peer']-r['own'] if r['peer_count'] else 0.
        base=[1.,r['own'],r['initial']-r['own'],r['round'],r['slot'],float(r['peer_count']>0)]
        if method!='self_dynamics':
            base += [gap]
        if method=='configuration_coupled':
            for configuration in configurations:
                base += [1.,r['own'],r['initial']-r['own'],r['round'],gap] if r['configuration']==configuration else [0.]*5
        matrix.append(base)
    return np.array(matrix,dtype=float)


def question_weights(rows):
    counts=defaultdict(int)
    for r in rows:
        counts[r['question']]+=1
    weights=np.array([1/counts[r['question']] for r in rows])
    return weights*len(rows)/weights.sum()


def ridge_fit(rows, method, alpha, configurations):
    x=design(rows,method,configurations); y=np.array([r['target'] for r in rows]); w=question_weights(rows)
    mean=np.average(x,axis=0,weights=w); scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
    scale[scale<1e-12]=1.;mean[0]=0.;scale[0]=1.
    z=(x-mean)/scale
    penalty=np.eye(x.shape[1])*alpha;penalty[0,0]=0.
    beta=np.linalg.solve((z.T*w)@z+penalty,(z.T*w)@y)
    raw=beta/scale;raw[0]-=float(np.sum(mean[1:]*raw[1:]))
    return dict(method=method,alpha=alpha,configurations=configurations,names=feature_names(method,configurations),
                coefficients=raw.tolist(),rank=int(np.linalg.matrix_rank(z)),features=x.shape[1],train_rows=len(rows))


def score(fitted, rows):
    if fitted['method'] in RIDGE_METHODS:
        return design(rows,fitted['method'],fitted['configurations'])@np.array(fitted['coefficients'])
    return np.array([fitted['intercept']+fitted['own']*r['own']+fitted['peer']*r['peer']+fitted['initial']*r['initial'] for r in rows])


def choose_alpha(rows, method, configurations):
    """Inner held-out questions; selection criterion is one-step question-macro MSE."""
    diagnostics=[]
    for alpha in ALPHAS:
        losses=[]
        for question in sorted({r['question'] for r in rows}):
            train=[r for r in rows if r['question']!=question]
            test=[r for r in rows if r['question']==question]
            fitted=ridge_fit(train,method,alpha,configurations)
            losses.append(float(np.mean((score(fitted,test)-[r['target'] for r in test])**2)))
        diagnostics.append(dict(alpha=alpha,inner_question_macro_mse=float(np.mean(losses))))
    best=min(diagnostics,key=lambda r:(r['inner_question_macro_mse'],-r['alpha']))
    return best['alpha'],diagnostics


def classical_fit(rows, method):
    own=np.array([r['own'] for r in rows]);peer=np.array([r['peer'] for r in rows]);y=np.array([r['target'] for r in rows]);w=question_weights(rows)
    params=dict(method=method,intercept=0.,own=1.,peer=0.,initial=0.,train_rows=len(rows))
    if method=='control_drift':
        x=np.column_stack([np.ones(len(rows)),own]);coeff=np.linalg.lstsq(x*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
        params.update(intercept=float(coeff[0]),own=float(coeff[1]))
    elif method=='degroot':
        gap=peer-own;denom=float(np.sum(w*gap**2));alpha=float(np.clip(np.sum(w*gap*(y-own))/denom,0,1)) if denom>1e-12 else 0.
        params.update(own=1-alpha,peer=alpha)
    elif method=='friedkin_johnsen':
        x=np.column_stack([own,peer,[r['initial'] for r in rows]])
        weights=simplex_fit(x*np.sqrt(w[:,None]),y*np.sqrt(w))
        params.update(own=float(weights[0]),peer=float(weights[1]),initial=float(weights[2]))
    return params


def identify(rows, methods=METHODS, subsets=('all_finite','valid_only')):
    fits=[];selection=[];predictions=[]
    questions=sorted({r['question'] for r in rows})
    # Vocabulary contains only known experimental settings, no target outcomes.
    configurations=sorted({r['configuration'] for r in rows})
    for heldout in questions:
        for subset in subsets:
            fitted_lookup={}
            for model in sorted({r['model'] for r in rows}):
                train=[r for r in rows if r['question']!=heldout and r['model']==model and (subset=='all_finite' or r['valid'])]
                for method in methods:
                    if method in RIDGE_METHODS:
                        alpha,diagnostics=choose_alpha(train,method,configurations)
                        fitted=ridge_fit(train,method,alpha,configurations)
                        schedules=sorted({r['schedule'] for r in rows})
                        selection.extend(dict(heldout_question=heldout,subset=subset,model=model,method=method,**d) for d in diagnostics)
                        fitted_lookup.update({(model,s,method):fitted for s in schedules})
                        fits.append(dict(heldout_question=heldout,subset=subset,model=model,schedule='joint',train_questions=sorted({r['question'] for r in train}),**fitted))
                    else:
                        for schedule in sorted({r['schedule'] for r in rows}):
                            selected=[r for r in train if r['schedule']==('no_peer' if method=='control_drift' else schedule)]
                            if not selected:
                                raise ValueError('Missing training condition')
                            fitted=classical_fit(selected,method)
                            fitted_lookup[(model,schedule,method)]=fitted
                            fits.append(dict(heldout_question=heldout,subset=subset,model=model,schedule=schedule,train_questions=sorted({r['question'] for r in selected}),**fitted))
            test=[r for r in rows if r['question']==heldout]
            for method in methods:
                for mode in ('one_step','recursive'):
                    state={(r['conversation'],r['prior_id']):r['own'] for r in test if r['round']==1}
                    for row in test:
                        inputs=dict(row)
                        if mode=='recursive':
                            inputs['own']=state[(row['conversation'],row['prior_id'])]
                            inputs['peer']=float(np.mean([state[(row['conversation'],pid)] for pid in row['peer_ids'].split('|')])) if row['peer_ids'] else inputs['own']
                        value=float(score(fitted_lookup[(row['model'],row['schedule'],method)],[inputs])[0])
                        if not np.isfinite(value):
                            raise ValueError('Nonfinite recursive prediction')
                        state[(row['conversation'],row['response_id'])]=value
                        if subset=='all_finite' or row['valid']:
                            predictions.append(dict(row,subset=subset,method=method,mode=mode,
                                predicted_margin=value,prediction=float(np.tanh(value/2)),target_margin=row['target'],
                                target_stance=float(np.tanh(row['target']/2))))
    return predictions,fits,selection


def metrics(predictions, group_keys):
    buckets=defaultdict(list)
    for r in predictions:
        buckets[tuple(r[k] for k in group_keys)].append(r)
    result=[]
    for values,selected in sorted(buckets.items()):
        per_question=[]
        for q in sorted({r['question'] for r in selected}):
            rr=[r for r in selected if r['question']==q]
            margin=np.array([r['predicted_margin']-r['target_margin'] for r in rr]);stance=np.array([r['prediction']-r['target_stance'] for r in rr])
            per_question.append([np.mean(abs(margin)),np.mean(margin**2),np.mean(abs(stance)),np.mean(stance**2)])
        means=np.mean(per_question,axis=0)
        result.append(dict(zip(group_keys,values),rows=len(selected),questions=len(per_question),margin_mae=float(means[0]),margin_rmse=float(np.sqrt(means[1])),stance_mae=float(means[2]),stance_rmse=float(np.sqrt(means[3]))))
    return result


def parameter_rows(fits):
    result=[]
    for fitted in fits:
        names=fitted.get('names',['intercept','own','peer','initial'])
        values=fitted.get('coefficients',[fitted.get(n,0.) for n in names])
        for name,value in zip(names,values):
            result.append(dict(heldout_question=fitted['heldout_question'],subset=fitted['subset'],model=fitted['model'],schedule=fitted['schedule'],method=fitted['method'],parameter=name,value=value))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/experiments/new_exp_1_a'))
    parser.add_argument('--output',type=Path,default=Path('results/modeling/new_exp_1_a_system_identification'))
    args=parser.parse_args();source=args.source.resolve();output=args.output.resolve()
    if source==output or source in output.parents or output in source.parents:
        parser.error('Output must be separate from source')
    if output.exists():
        parser.error('Choose a fresh output directory')
    before=hashes(source)
    rows=extract(source)
    print(f'Identifying {len(rows)} updates across {len({r["question"] for r in rows})} held-out question folds',flush=True)
    predictions,fits,selection=identify(rows)
    actual=observations(source);agents,groups=outcome_rows(predictions,actual)
    if hashes(source)!=before:
        raise RuntimeError('Source changed during identification')
    output.mkdir(parents=True)
    export(output,'modeling_rows',rows);export(output,'predictions',predictions)
    # Nested lists stay in JSON; coefficient table below supplies a flat CSV.
    (output/'fitted_models.json').write_text(json.dumps(fits,indent=2,allow_nan=False)+'\n')
    export(output,'parameters',parameter_rows(fits));export(output,'regularization_selection',selection)
    base=('schedule','subset','method','mode')
    for name,keys in [('summary',base),('by_configuration',base+('composition',)),('by_agent',base+('model','slot')),
                      ('by_round',base+('round',)),('by_question',base+('question','composition'))]:
        export(output,name,metrics(predictions,keys))
    export(output,'agent_outcomes',agents);export(output,'group_outcomes',groups)
    export(output,'agent_outcome_summary',outcome_summary(agents,'agent'));export(output,'group_outcome_summary',outcome_summary(groups,'group'))
    last={}
    for r in rows:last[r['conversation']]=max(last.get(r['conversation'],0),r['round'])
    export(output,'agent_final',outcome_summary([r for r in agents if r['round']==last[r['conversation']]],'agent'))
    export(output,'group_final',outcome_summary([r for r in groups if r['round']==last[r['conversation']]],'group'))
    export(output,'configuration_outcomes',configuration_outcomes(agents,groups,last))
    export(output,'parameter_stability',stability(fits))
    export(output,'residual_diagnostics',residual_diagnostics(predictions))
    events=transition_events(predictions,actual)
    export(output,'transition_events',events)
    export(output,'transition_summary',transition_summary(events))
    export(output,'observed_configuration_comparison',observed_comparison(rows,actual))
    (output/'manifest.json').write_text(json.dumps(dict(source=str(source),source_sha256=before,source_unchanged=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependencies_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('model_saved_pilot.py','evaluate_saved_outcomes.py','system_identification_figures.py')},
        python=sys.version,numpy=np.__version__,target='post-turn raw likelihood margin; stance=tanh(margin/2)',
        methods=METHODS,alphas=ALPHAS,validation='outer leave-one-question-out; inner leave-one-question-out alpha selection by one-step margin MSE',
        weighting='equal weight per question; feature centering/scaling fitted only on training folds',
        ridge_structure='checkpoint-specific shared dynamics; configuration_coupled adds ridge-regularized composition-and-slot feature adjustments',
        sensitivity='valid_only fits/scores valid current, prior, initial, delivered-peer responses; rollout still advances all states',
        limitations=['Ten previously inspected questions; exploratory internal validation, not an untouched test set.',
            'Ridge tuning optimizes one-step margin error, not recursive or answer accuracy.',
            'Configuration contrasts and coefficients are descriptive; redundant features make individual ridge coefficients non-unique across parameterizations.',
            'Cross-checkpoint margins are not calibrated; compare within-checkpoint diagnostics too.',
            'Raw margin dynamics are unbounded; no clipping hides recursive instability.',
            'Actual message identities/timing are used, but semantic content and future messages are not modeled.',
            'Classical reference models are refitted on margins, unlike the previous stance-target analysis.',
            'Initial states come from observed initial responses; question text alone is not sufficient input.']),indent=2)+'\n')
    from system_identification_figures import generate
    generate(output,rows,predictions,fits,actual)
    print(output,flush=True)


def configuration_outcomes(agents,groups,last):
    result=[]
    for level,records in [('agent',agents),('group',groups)]:
        for composition in sorted({r['composition'] for r in records}):
            selected=[r for r in records if r['composition']==composition and r['round']==last[r['conversation']]]
            result.extend(dict(composition=composition,level=level,**r) for r in outcome_summary(selected,level))
    # Agent/group schemas differ, so normalize CSV fields.
    names=list(dict.fromkeys(k for r in result for k in r))
    return [{k:r.get(k) for k in names} for r in result]


def stability(fits):
    buckets=defaultdict(list)
    for r in parameter_rows(fits):
        buckets[(r['subset'],r['model'],r['schedule'],r['method'],r['parameter'])].append(r['value'])
    return [dict(zip(('subset','model','schedule','method','parameter'),key),mean=float(np.mean(values)),std=float(np.std(values)),minimum=float(min(values)),maximum=float(max(values))) for key,values in sorted(buckets.items())]


def residual_diagnostics(predictions):
    buckets=defaultdict(list)
    for r in predictions:
        if r['mode']=='one_step':buckets[(r['schedule'],r['subset'],r['method'],r['composition'],r['model'])].append(r)
    result=[]
    for key,rows in sorted(buckets.items()):
        residual={(r['conversation'],r['response_id']):r['target_margin']-r['predicted_margin'] for r in rows}
        pairs=[(residual[(r['conversation'],r['prior_id'])],residual[(r['conversation'],r['response_id'])]) for r in rows if (r['conversation'],r['prior_id']) in residual]
        correlation=float(np.corrcoef(np.array(pairs).T)[0,1]) if len(pairs)>2 and np.min(np.std(pairs,axis=0))>1e-12 else None
        result.append(dict(zip(('schedule','subset','method','composition','model'),key),rows=len(rows),mean_residual=float(np.mean(list(residual.values()))),lag1_pairs=len(pairs),lag1_correlation=correlation))
    return result


def transition_events(predictions,actual):
    predicted={(r['conversation'],r['response_id'],r['subset'],r['method'],r['mode']):r['predicted_margin'] for r in predictions}
    result=[]
    for r in predictions:
        prior=actual[(r['conversation'],r['prior_id'])];current=actual[(r['conversation'],r['response_id'])]
        valid=prior['explicit'] in ('Yes','No') and current['explicit'] in ('Yes','No')
        if r['mode']=='one_step':
            previous=prior['explicit']
        else:
            key=(r['conversation'],r['prior_id'],r['subset'],r['method'],r['mode'])
            if r['round']>1 and key not in predicted:
                # Sensitivity eligibility can remove the previous prediction from exports.
                # Do not manufacture a previous predicted answer.
                previous=None
            else:
                value=predicted[key] if r['round']>1 else r['initial']
                previous='Yes' if value>0 else 'No' if value<0 else 'Tie'
        following='Yes' if r['predicted_margin']>0 else 'No' if r['predicted_margin']<0 else 'Tie'
        label=current['label'];labeled=label in ('Yes','No')
        result.append(dict(question=r['question'],conversation=r['conversation'],model=r['model'],composition=r['composition'],schedule=r['schedule'],round=r['round'],subset=r['subset'],method=r['method'],mode=r['mode'],
            actual_reversal=(prior['explicit']!=current['explicit']) if valid else None,
            predicted_reversal=(previous!=following) if previous in ('Yes','No') and following in ('Yes','No') else None,
            actual_correct_to_incorrect=(prior['explicit']==label and current['explicit']!=label) if valid and labeled else None,
            predicted_correct_to_incorrect=(previous==label and following!=label) if previous in ('Yes','No') and following in ('Yes','No') and labeled else None,
            actual_incorrect_to_correct=(prior['explicit']!=label and current['explicit']==label) if valid and labeled else None,
            predicted_incorrect_to_correct=(previous!=label and following==label) if previous in ('Yes','No') and following in ('Yes','No') and labeled else None))
    return result


def transition_summary(events):
    from evaluate_saved_outcomes import macro
    buckets=defaultdict(list)
    for r in events:buckets[(r['schedule'],r['subset'],r['method'],r['mode'],r['composition'])].append(r)
    result=[]
    for key,rr in sorted(buckets.items()):
        for event in ('reversal','correct_to_incorrect','incorrect_to_correct'):
            valid=[r for r in rr if r['actual_'+event] is not None and r['predicted_'+event] is not None]
            result.append(dict(zip(('schedule','subset','method','mode','composition'),key),event=event,eligible_rows=len(valid),
                actual_rate=macro(valid,lambda r:r['actual_'+event]),predicted_rate=macro(valid,lambda r:r['predicted_'+event]),
                detection_accuracy=macro(valid,lambda r:r['actual_'+event]==r['predicted_'+event]),
                precision=macro(valid,lambda r:r['actual_'+event] if r['predicted_'+event] else None),
                recall=macro(valid,lambda r:r['predicted_'+event] if r['actual_'+event] else None)))
    return result


def observed_comparison(rows,actual):
    from evaluate_saved_outcomes import group_answer
    buckets=defaultdict(list)
    for r in rows:buckets[(r['question'],r['composition'],r['schedule'])].append(r)
    result=[]
    for key,rr in sorted(buckets.items()):
        final=max(r['round'] for r in rr);ends=[actual[(r['conversation'],r['response_id'])] for r in rr if r['round']==final]
        starts=[actual[(r['conversation'],r['prior_id'])] for r in rr if r['round']==1]
        byconversation=defaultdict(list)
        for r in ends:byconversation[r['conversation']].append(r)
        groups=[group_answer([r['explicit'] for r in group]) for group in byconversation.values()]
        result.append(dict(zip(('question','composition','schedule'),key),label=ends[0]['label'],initial_correct=sum(r['explicit']==r['label'] for r in starts)/len(starts),final_correct=sum(r['explicit']==r['label'] for r in ends)/len(ends),
            initial_valid=sum(r['explicit'] in ('Yes','No') for r in starts),final_valid=sum(r['explicit'] in ('Yes','No') for r in ends),agent_outputs=len(ends),consensus=sum(g in ('Yes','No') for g in groups)/len(groups),mean_absolute_margin_update=float(np.mean([abs(r['target']-r['own']) for r in rr]))))
    return result


if __name__=='__main__':main()
