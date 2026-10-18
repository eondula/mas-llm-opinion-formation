"""Descriptive StrategyQA behavior from recorded states and actual message deliveries.

Peer movement is an association, not a causal estimate. No model calls are made.
"""
from .output_state import measurement
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from .storage import write
from .trajectories import conversations

VERSION = 'strategyqa-behavioral-v2'


def average(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def fraction(values):
    return average([int(v) for v in values if v is not None])


def number(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def delta(a, b):
    return a-b if a is not None and b is not None else None


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(k for row in rows for k in row)) if rows else ['task'])
        writer.writeheader()
        writer.writerows(rows)


def condition(config):
    return dict(models=' | '.join(config['models']), agents=config['agents'],
                schedule=config['schedule'],
                initialization=json.dumps(config['initialization']),
                decoding=config['decoding'])


def analyze(path, tolerance=0.01):
    path = Path(path)
    records = [(file,record) for file,record in conversations(path) if record.get('dataset')]
    if not records:
        raise ValueError('No recorded StrategyQA conversations found')
    turns, agents, groups, delivery_rows = [], [], [], []
    for file, record in records:
        if not record.get('dataset'):
            continue
        config = record['config']
        identity = dict(protocol_hash=record.get('protocol_hash'),pair_id=record.get('pair_id'),initialization_hash=record.get('initialization_hash'),control_design=record.get('control_design','independently_initialized'),task=file.stem, question_id=record['question_id'],
                        repetition=record.get('repetition', 0), research_split=record.get('research_split'),
                        source_partition=record['dataset']['partition'], stance_measurement='decision_time' if config.get('response_protocol')=='decision_time_v2' else 'post_response_probe', **condition(config))
        expected = record.get('evaluation', {}).get('reference_answer')
        labeled = expected in ('Yes', 'No')
        rows = record['rows']
        by_id = {r['response_id']: r for r in rows}
        positions = {r['response_id']: i for i,r in enumerate(rows)}
        by_agent = defaultdict(list)
        for row in rows:
            by_agent[row['agent']].append(row)
        for name, history in by_agent.items():
            history.sort(key=lambda r: r['round'])
            previous = None
            previous_direction = None
            local = []
            for row in history:
                stance = number(measurement(row).get('stance'))
                margin = number(measurement(row).get('margin'))
                old_stance = number(measurement(previous).get('stance')) if previous else None
                old_margin = number(measurement(previous).get('margin')) if previous else None
                change = delta(stance, old_stance)
                direction = (1 if change > tolerance else -1 if change < -tolerance else 0) if change is not None else None
                valid = row.get('answer') in ('Yes', 'No')
                old_valid = previous and previous.get('answer') in ('Yes', 'No')
                sources = []
                missing = 0
                for delivery in row.get('deliveries', []):
                    source = by_id.get(delivery.get('source_id'))
                    if source is None or source['agent'] != delivery['speaker'] or source['round'] != delivery['source_round'] or positions[source['response_id']] >= positions[row['response_id']]:
                        missing += 1
                    else:
                        sources.append(source)
                peer_stance = average([number(measurement(r).get('stance')) for r in sources])
                peers_valid = [r['answer'] for r in sources if r.get('answer') in ('Yes', 'No')]
                peer_majority = None
                if len(peers_valid) == len(sources) and peers_valid and peers_valid.count('Yes') != peers_valid.count('No'):
                    peer_majority = 'Yes' if peers_valid.count('Yes') > peers_valid.count('No') else 'No'
                toward = None
                distance_reduction = None
                if old_stance is not None and stance is not None and peer_stance is not None and not missing:
                    toward = (1 if peer_stance > old_stance else -1 if peer_stance < old_stance else 0) * change
                    distance_reduction = abs(old_stance-peer_stance)-abs(stance-peer_stance)
                correct = row['answer'] == expected if labeled and valid else None
                old_correct = previous.get('answer') == expected if labeled and old_valid else None
                transition = (('correct' if old_correct else 'incorrect')+'_to_'+('correct' if correct else 'incorrect')) if old_correct is not None and correct is not None else None
                for delivery in row.get('deliveries', []):
                    source=by_id.get(delivery.get('source_id'))
                    resolved=source in sources
                    delivery_rows.append(dict(**identity,agent=name,round=row['round'],response_id=row['response_id'],
                        speaker=delivery['speaker'],source_response_id=delivery.get('source_id'),source_round=delivery.get('source_round'),resolved=resolved,
                        source_stance=number(measurement(source).get('stance')) if resolved else None,
                        source_answer=source.get('answer') if resolved else None,
                        exact_peer_argument=row.get('argument','').strip()==source.get('argument','').strip() if resolved and valid and source.get('answer') in ('Yes','No') else None))
                point = dict(**identity, agent=name, model=row.get('model'), round=row['round'],
                             response_id=row['response_id'],previous_response_id=previous['response_id'] if previous else None,previous_stance=old_stance,
                             exact_delivered_peer_argument=any(row.get('argument','').strip()==source.get('argument','').strip() for source in sources if source.get('answer') in ('Yes','No')) if valid and sources and not missing else None,
                             answer=row.get('answer'), valid=valid, labeled=labeled, correct=correct,
                             argument_words=len(row.get('argument','').split()), sentiment=number(row.get('sentiment')),
                             adjacent_message_cosine=row.get('message_similarity',{}).get('adjacent'),
                             initial_message_cosine=row.get('message_similarity',{}).get('initial'),
                             answer_oscillation=(row['answer']==local[-2]['answer'] and row['answer']!=previous['answer']) if len(local)>1 and valid and old_valid and local[-2]['valid'] else None,
                             forced=row.get('forced', False), stance=stance, margin=margin,
                             stance_change=change, absolute_stance_change=abs(change) if change is not None else None,
                             margin_change=delta(margin, old_margin),
                             stance_strength_change=delta(abs(stance) if stance is not None else None, abs(old_stance) if old_stance is not None else None),
                             answer_reversal=(row['answer'] != previous['answer']) if old_valid and valid else None,
                             correctness_transition=transition,
                             stable=abs(change) <= tolerance if change is not None else None,
                             direction_reversal=direction * previous_direction < 0 if direction is not None and previous_direction is not None else None,
                             repeated_argument=row.get('argument','').strip() == previous.get('argument','').strip() if previous and valid and old_valid else None,
                             delivered_source_ids=json.dumps([d.get('source_id') for d in row.get('deliveries',[])]),delivered_source_rounds=json.dumps([d.get('source_round') for d in row.get('deliveries',[])]),delivered_peer_stances=json.dumps([number(measurement(s).get('stance')) for s in sources]),delivered_peers=len(row.get('deliveries', [])), unresolved_deliveries=missing,
                             peer_mean_stance=peer_stance, peer_majority=peer_majority,
                             peer_answer_disagreement=fraction([a != previous['answer'] for a in peers_valid]) if old_valid and peers_valid else None,
                             movement_toward_peers=toward, peer_distance_reduction=distance_reduction,
                             matches_peer_majority=row['answer'] == peer_majority if valid and peer_majority else None,
                             delivery_stance_change=delta(number(row.get('decision',{}).get('stance')),number(row.get('before',{}).get('stance'))),
                             response_stance_change=delta(stance,number(row.get('decision',{}).get('stance'))))
                local.append(point); turns.append(point)
                previous = row
                previous_direction = direction
            first, last = local[0], local[-1]
            agents.append(dict(**identity, agent=name, model=first['model'], labeled=labeled,
                               initial_answer=first['answer'], final_answer=last['answer'],
                               initial_valid=first['valid'], final_valid=last['valid'],
                               initial_correct=first['correct'], final_correct=last['correct'],
                               initial_stance=first['stance'], final_stance=last['stance'],
                               net_stance_change=delta(last['stance'],first['stance']),
                               mean_absolute_stance_change=average([r['absolute_stance_change'] for r in local]),
                               mean_adjacent_message_cosine=average([r['adjacent_message_cosine'] for r in local]),
                               final_initial_message_cosine=last['initial_message_cosine'],
                               responses=len(local), invalid_responses=sum(not r['valid'] for r in local),
                               answer_reversals=sum(r['answer_reversal'] is True for r in local),
                               valid_answer_transitions=sum(r['answer_reversal'] is not None for r in local),
                               direction_reversals=sum(r['direction_reversal'] is True for r in local),
                               answer_oscillations=sum(r['answer_oscillation'] is True for r in local),
                               repeated_arguments=sum(r['repeated_argument'] is True for r in local),
                               stable_update_fraction=fraction([r['stable'] for r in local]),
                               correct_to_incorrect=sum(r['correctness_transition']=='correct_to_incorrect' for r in local),
                               incorrect_to_correct=sum(r['correctness_transition']=='incorrect_to_correct' for r in local)))
        local_groups=[]
        rounds=sorted({r['round'] for r in rows})
        for t in rounds:
            current=[r for r in rows if r['round']==t]
            yes=sum(r.get('answer')=='Yes' for r in current)
            no=sum(r.get('answer')=='No' for r in current)
            complete=len(current)==config['agents'] and yes+no==config['agents']
            values=[number(measurement(r).get('stance')) for r in current]
            values=[v for v in values if v is not None]
            majority=('Yes' if yes>no else 'No') if complete and yes!=no else None
            consensus=complete and (yes==len(current) or no==len(current)) if config['agents']>1 else None
            group=dict(**identity, round=t, initial=t==rounds[0], final=t==rounds[-1],
                       labeled=labeled, yes=yes, no=no, invalid=len(current)-yes-no,
                       agent_accuracy=sum(r.get('answer')==expected for r in current)/config['agents'] if labeled else None,
                       mean_adjacent_message_cosine=average([r.get('message_similarity',{}).get('adjacent') for r in current]),
                       mean_initial_message_cosine=average([r.get('message_similarity',{}).get('initial') for r in current]),
                       consensus=consensus, disagreement=(yes>0 and no>0) if complete and config['agents']>1 else None,
                       pairwise_answer_disagreement=yes*no/(len(current)*(len(current)-1)/2) if complete and len(current)>1 else None,
                       majority=majority, majority_correct=majority==expected if majority and labeled else None,
                       consensus_correct=majority==expected if consensus and labeled else None,
                       mean_stance=average(values), stance_dispersion=pstdev(values) if len(values)==config['agents'] else None,
                       stance_range=max(values)-min(values) if len(values)==config['agents'] else None)
            local_groups.append(group)
        first_consensus=next((r['round'] for r in local_groups if r['consensus']),None)
        for group in local_groups:
            group['first_consensus_round']=first_consensus
            group['first_consensus_correct']=next((r['consensus_correct'] for r in local_groups if r['consensus']),None)
            groups.append(group)
    buckets=defaultdict(lambda: dict(turns=[],agents=[],groups=[]))
    for kind,data in [('turns',turns),('agents',agents),('groups',groups)]:
        for row in data:
            key=tuple(row[k] for k in ('models','agents','schedule','initialization','decoding','research_split','source_partition'))
            buckets[key][kind].append(row)
    summaries=[]
    for key,data in sorted(buckets.items(),key=lambda item:str(item[0])):
        ts, ats, gs=data['turns'],data['agents'],data['groups']
        initials=[r for r in gs if r['initial']];finals=[r for r in gs if r['final']]
        initial_labeled=[r for r in ats if r['labeled']];final_labeled=initial_labeled
        label_groups=[r for r in finals if r['labeled']]
        summaries.append(dict(zip(('models','agents','schedule','initialization','decoding','research_split','source_partition'),key),
                              questions=len({r['question_id'] for r in ats}),runs=len({r['repetition'] for r in ats}),
                              conversations=len(finals), responses=len(ts), invalid_responses=sum(not r['valid'] for r in ts),
                              labeled_agent_trajectories=len(initial_labeled),
                              initial_accuracy=fraction([r['initial_correct'] is True for r in initial_labeled]),
                              final_accuracy=fraction([r['final_correct'] is True for r in final_labeled]),
                              initial_disagreement_rate=fraction([r['disagreement'] for r in initials]),
                              final_consensus_rate=fraction([r['consensus'] for r in finals]),
                              initial_majority_accuracy=fraction([r['majority_correct'] is True for r in initials if r['labeled']]),
                              final_consensus_accuracy=fraction([r['consensus_correct'] for r in finals if r['consensus']]),
                              correct_final_consensus_rate=fraction([r['consensus_correct'] is True for r in finals if r['labeled'] and r['agents']>1]),
                              exact_peer_argument_reuse_rate=fraction([r['exact_delivered_peer_argument'] for r in ts]),
                              final_majority_accuracy=fraction([r['majority_correct'] is True for r in label_groups]),
                              mean_final_dispersion=average([r['stance_dispersion'] for r in finals]),
                              mean_absolute_stance_change=average([r['absolute_stance_change'] for r in ts]),
                              answer_reversals=sum(r['answer_reversals'] for r in ats),
                              answer_reversal_rate=sum(r['answer_reversals'] for r in ats)/sum(r['valid_answer_transitions'] for r in ats) if sum(r['valid_answer_transitions'] for r in ats) else None,
                              correct_to_incorrect=sum(r['correct_to_incorrect'] for r in ats),
                              incorrect_to_correct=sum(r['incorrect_to_correct'] for r in ats),
                              direction_reversals=sum(r['direction_reversals'] for r in ats),
                              answer_oscillations=sum(r['answer_oscillations'] for r in ats),
                              mean_first_consensus_round=average([r['first_consensus_round'] for r in finals]),
                              final_disagreement_rate=fraction([r['disagreement'] for r in finals]),
                              mean_argument_words=average([r['argument_words'] for r in ts]),
                              mean_adjacent_message_cosine=average([r['adjacent_message_cosine'] for r in ts]),
                              mean_final_initial_message_cosine=average([r['final_initial_message_cosine'] for r in ats]),
                              stable_update_fraction=fraction([r['stable'] for r in ts]),
                              repeated_arguments=sum(r['repeated_arguments'] for r in ats),
                              mean_peer_distance_reduction=average([r['peer_distance_reduction'] for r in ts]),
                              mean_movement_toward_peers=average([r['movement_toward_peers'] for r in ts])))
    analysis=path/'analysis'
    analysis.mkdir(exist_ok=True)
    definitions=dict(version=VERSION,stance_stability_tolerance=tolerance,
                     accuracy_denominator='All labeled agent trajectories, including invalid outputs as unsuccessful',
                     majority_accuracy_denominator='All labeled conversations; ties and invalid groups count as unsuccessful',
                     aggregation='Pooled observations within each model/group/schedule/start/decoding/dataset split condition; questions and runs reported separately',
                     consensus='All configured agents have valid identical answers; undefined for a single agent',
                     peer_movement='Descriptive association with source-row primary stance (decision-time when recorded, legacy post-response otherwise) for messages actually delivered; not causal influence',
                     oscillation='Sign reversal between consecutive stance updates exceeding the stability tolerance; flat updates break the sequence',
                     missing_values='Null means unavailable or inapplicable; invalid outputs are not replaced with scored answers')
    write(analysis/'behavioral.json',dict(definitions=definitions,summary=summaries,turns=turns,agents=agents,groups=groups))
    write(path/'summary.json',dict(definitions=definitions,rows=summaries))
    for name,data in [('agent_turns',turns),('agent_trajectories',agents),('group_dynamics',groups),('peer_deliveries',delivery_rows),('summary',summaries)]:
        csv_write(path/'tables'/f'{name}.csv',data)
    match_fields=('pair_id','protocol_hash','question_id','repetition','agent','models','agents','initialization','decoding','research_split','source_partition')
    controls={tuple(r[k] for k in match_fields):r for r in agents if r['schedule']=='no_peer'}
    contrasts=[]
    for row in agents:
        control=controls.get(tuple(row[k] for k in match_fields))
        if row['schedule']=='no_peer' or control is None:
            continue
        contrasts.append(dict(**{k:row[k] for k in match_fields},schedule=row['schedule'],
                              deliberation_task=row['task'],control_task=control['task'],control_design=row['control_design'],initialization_identical=row['initialization_hash']==control['initialization_hash'] if row['pair_id'] else None,
                              final_stance_difference=delta(row['final_stance'],control['final_stance']),
                              final_correctness_difference=int(row['final_correct'] is True)-int(control['final_correct'] is True) if row['labeled'] and control['labeled'] else None))
    csv_write(path/'tables/no_peer_comparisons.csv',contrasts)
    (path/'RESULTS.md').write_text(
        'Experiment outputs\n\n'
        'summary.json and tables/summary.csv contain condition summaries. '
        'analysis/behavioral.json and tables/ contain per-turn, agent and group metrics. '
        'tables/no_peer_comparisons.csv pairs available controls by question, repetition and agent. '
        'Conversations and raw measurements are in conversations/; figures are in figures/. '
        'Inputs, source hashes and environment are saved alongside these outputs. '
        'Peer movement and control differences are descriptive, not proof of causal influence.\n')
    return summaries


def plots(path):
    """Save aggregate plots; the experiment page displays only summary rows."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((path/'analysis/behavioral.json').read_text())
    grouped=defaultdict(list)
    for row in data['groups']:
        grouped[(row['schedule'],row['initialization'],row['models'],row['agents'],row['research_split'],row['round'])].append(row)
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    curves=defaultdict(list)
    for key,rows in grouped.items():
        curves[key[:-1]].append((key[-1],average([r['stance_dispersion'] for r in rows]),fraction([r['consensus'] for r in rows]),average([r['agent_accuracy'] for r in rows])))
    for key,points in curves.items():
        points.sort(); label=f'{key[0]} | {key[1]} | {key[3]} agents | {key[4]} | {key[2]}'
        axes[0].plot([p[0] for p in points],[p[1] if p[1] is not None else float('nan') for p in points],marker='o',label=label)
        axes[1].plot([p[0] for p in points],[p[2] if p[2] is not None else float('nan') for p in points],marker='o',label=label)
        axes[2].plot([p[0] for p in points],[p[3] if p[3] is not None else float('nan') for p in points],marker='o',label=label)
    round_ticks=sorted({key[-1] for key in grouped})
    for ax,title in zip(axes,['Stance dispersion','Consensus fraction','Agent accuracy']):
        ax.set(xlabel='Round',ylabel=title,title=title)
        ax.set_xticks(round_ticks)
        if curves:ax.legend(fontsize=6)
    axes[1].set_ylim(-.05,1.05);axes[2].set_ylim(-.05,1.05)
    fig.tight_layout();(path/'figures').mkdir(exist_ok=True)
    fig.savefig(path/'figures/behavioral_overview.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    similarity_curves=defaultdict(list)
    for key,rows in grouped.items():
        similarity_curves[key[:-1]].append((key[-1],average([r['mean_adjacent_message_cosine'] for r in rows]),average([r['mean_initial_message_cosine'] for r in rows])))
    for key,points in similarity_curves.items():
        points.sort();label=f'{key[0]} | {key[1]} | {key[3]} agents | {key[4]} | {key[2]}'
        ax.plot([p[0] for p in points],[p[1] if p[1] is not None else float('nan') for p in points],marker='o',label=label+' · previous round')
        ax.plot([p[0] for p in points],[p[2] if p[2] is not None else float('nan') for p in points],linestyle='--',marker='x',label=label+' · initial round')
    ax.set_xticks(round_ticks)
    ax.set(xlabel='Round',ylabel='Cosine similarity',title='Within-agent public message similarity',ylim=(-1.05,1.05))
    if similarity_curves:ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(path/'figures/message_similarity.png',dpi=160);plt.close(fig)
