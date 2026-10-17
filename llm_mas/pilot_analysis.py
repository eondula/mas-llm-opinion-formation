"""Paired-control validation and descriptive variation diagnostics; no weight fitting."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .storage import write,read
from .config import digest
from .trajectories import conversations
from .behavioral import csv_write


def analyze(path,data):
    path=Path(path);pairs=defaultdict(list)
    for _,record in conversations(path):
        if record.get('pair_id'):pairs[record['pair_id']].append(record)
    checks=[]
    for pair,records in pairs.items():
        snapshot=read(path/'initializations'/f'{pair}.json')
        starts=[digest([{k:r[k] for k in original} for r,original in zip([r for r in record['rows'] if r['round']==0],snapshot['rows'])]) for record in records]
        schedules={r['config']['schedule'] for r in records}
        valid=len(records)==2 and 'no_peer' in schedules and len(set(starts))==1 and all(h==r['initialization_hash'] for h,r in zip(starts,records))
        for record in records:
            for row in record['rows']:
                if record['config']['schedule']=='no_peer' and row.get('deliveries'):valid=False
                if record['config']['schedule']=='synchronous' and row['round'] and any(d['source_round']!=row['round']-1 for d in row['deliveries']):valid=False
        checks.append(dict(pair_id=pair,branches=len(records),initialization_identical=len(set(starts))==1,valid=valid))
    csv_write(path/'tables/pairing_validation.csv',checks)
    paired_outcomes=[]
    for pair in pairs:
        members=[r for r in data['groups'] if r.get('pair_id')==pair and r['final']]
        control=next((r for r in members if r['schedule']=='no_peer'),None)
        discussion=next((r for r in members if r['schedule']!='no_peer'),None)
        if control and discussion:
            paired_outcomes.append(dict(pair_id=pair,question_id=discussion['question_id'],repetition=discussion['repetition'],models=discussion['models'],
                discussion_majority=discussion.get('majority'),control_majority=control.get('majority'),
                agent_accuracy_difference=discussion['agent_accuracy']-control['agent_accuracy'] if discussion['agent_accuracy'] is not None and control['agent_accuracy'] is not None else None,
                majority_correctness_difference=int(discussion.get('majority_correct') is True)-int(control.get('majority_correct') is True),
                consensus_difference=int(discussion['consensus'] is True)-int(control['consensus'] is True)))
    csv_write(path/'tables/paired_group_outcomes.csv',paired_outcomes)
    initial={(r['task'],r['agent']):r['stance'] for r in data['turns'] if r['round']==0}
    transitions=[]
    for r in data['turns']:
        if not r['round']:continue
        transitions.append(dict(r,initial_stance=initial[(r['task'],r['agent'])],own_peer_gap=(r['peer_mean_stance']-r['previous_stance']) if r['peer_mean_stance'] is not None and r['previous_stance'] is not None else None))
    csv_write(path/'tables/model_transitions.csv',transitions)
    initial_groups={r['task']:r for r in data['groups'] if r['initial']}
    strata=defaultdict(list)
    for r in transitions:
        state=initial_groups[r['task']].get('disagreement')
        strength='weak' if r['initial_stance'] is not None and abs(r['initial_stance'])<.5 else 'strong' if r['initial_stance'] is not None else 'missing'
        strata[(r['models'],r['schedule'],state,strength)].append(r)
    from .behavioral import average,fraction
    strata_rows=[dict(models=k[0],schedule=k[1],initial_disagreement=k[2],initial_strength=k[3],updates=len(v),mean_absolute_stance_change=average([r['absolute_stance_change'] for r in v]),reversal_fraction=fraction([r['answer_reversal'] for r in v])) for k,v in strata.items()]
    csv_write(path/'tables/initial_state_strata.csv',strata_rows)
    diagnostics=[]
    for model in sorted({r['model'] for r in transitions}):
        subset=[r for r in transitions if r['model']==model and all(r[k] is not None for k in ('initial_stance','previous_stance','peer_mean_stance','stance_change'))]
        x=np.array([[1,r['initial_stance'],r['previous_stance'],r['peer_mean_stance']] for r in subset],dtype=float)
        rank=int(np.linalg.matrix_rank(x)) if len(x) else 0
        diagnostics.append(dict(model=model,updates=len(subset),feature_columns=4,feature_rank=rank,full_rank=rank==4,
            condition_number=float(np.linalg.cond(x)) if rank==4 else None,
            update_sd=float(np.std([r['stance_change'] for r in subset])) if subset else None,
            own_peer_gap_sd=float(np.std([r['own_peer_gap'] for r in subset])) if subset else None))
    csv_write(path/'tables/informative_variation.csv',diagnostics)
    result=dict(pairing=checks,paired_group_outcomes=paired_outcomes,initial_state_strata=strata_rows,variation=diagnostics,note='Rank and variation diagnose whether future influence-weight estimation is informative. No influence weights are fitted. Cross-model stance scales remain uncalibrated.')
    write(path/'analysis/pilot_validation.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (path/'figures').mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for model in sorted({r['model'] for r in transitions}):
        subset=[r for r in transitions if r['model']==model]
        for ax,field in zip(axes,['initial_stance','own_peer_gap','stance_change']):
            values=[r[field] for r in subset if r[field] is not None]
            if values:ax.hist(values,bins=15,alpha=.5,label=model.split('/')[-1])
    for ax,title in zip(axes,['Initial stance (updates weighted)','Delivered peer minus previous stance','Stance updates']):
        ax.set_title(title);ax.set_ylabel('Updates')
        if ax.has_data():ax.legend(fontsize=6)
    for ext in ('png','pdf'):fig.savefig(path/f'figures/informative_variation.{ext}',dpi=150)
    plt.close(fig)
    if any(not r['valid'] for r in checks):raise ValueError('Paired-control validation failed; inspect tables/pairing_validation.csv')
    return result
