"""Leave-one-conversation-out regression, with persistence as a baseline."""
from .trajectories import read_conversation
import json
import numpy as np
from .storage import write


def run(path):
    examples=[]
    for file in sorted((path/'conversations').glob('*.json')):
        record=read_conversation(file)
        for row in record['rows']:
            if not row['decision'] or row['round']==0:continue
            examples.append(dict(task=file.stem,agent=row['agent'],model=row['model'],round=row['round'],
                before=row['before']['margin'],target=row['decision']['margin'],
                peer_count=len(row.get('deliveries',[])),history_length=len(row['decision_history'])))
    predictions=[]
    for model in sorted({r['model'] for r in examples}):
        selected=[r for r in examples if r['model']==model]
        tasks=sorted({r['task'] for r in selected})
        if len(tasks)<2:continue
        for task in tasks:
            train=[r for r in selected if r['task']!=task];test=[r for r in selected if r['task']==task]
            def features(rows):return np.array([[r['before'],r['peer_count'],r['history_length']] for r in rows],dtype=float)
            x=features(train);mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale==0]=1
            design=np.column_stack([np.ones(len(x)),(x-mean)/scale]);y=np.array([r['target'] for r in train])
            penalty=np.eye(design.shape[1]);penalty[0,0]=0
            coefficients=np.linalg.solve(design.T@design+penalty,design.T@y)
            values=np.column_stack([np.ones(len(test)),(features(test)-mean)/scale])@coefficients
            for r,value in zip(test,values):predictions.append(dict(r,prediction=float(value),baseline=r['before'],train_tasks=[t for t in tasks if t!=task]))
    write(path/'analysis/prediction.json',dict(method='Leave-one-conversation-out ridge (alpha=1); model-specific; next decision prediction, not autonomous trajectory prediction',
         status='complete' if predictions else 'insufficient_conversations',rows=predictions))


def predict_groups(path):
    """Use separate factorial probes, never fit on target conversation outcomes."""
    source=path/'measurements/history.json'
    if not source.exists():return
    factors=json.loads(source.read_text());coefficients={}
    for key in {(r['model'],r['scenario']) for r in factors}:
        rows=[r for r in factors if (r['model'],r['scenario'])==key]
        x=np.array([[1,1 if r['self_strategy']==1 else -1,1 if r['peer_strategy']==1 else -1,(1 if r['self_strategy']==r['peer_strategy'] else -1)] for r in rows])
        if len(rows)==4:coefficients[key]=np.linalg.solve(x,np.array([r['after']['margin'] for r in rows]))
    results=[]
    for source in sorted((path/'conversations').glob('*.json')):
        record=read_conversation(source);rows=record['rows'];c=record['config'];initial=[r for r in rows if r['round']==0]
        names=[r['agent'] for r in initial];models={r['agent']:r['model'] for r in initial};state={r['agent']:r['strategy'] for r in initial}
        if len(names)!=2 or c['schedule']=='no_peer':continue
        if any((models[n],record['scenario']['id']) not in coefficients for n in names):continue
        for t in range(1,c['rounds']+1):
            old=dict(state);order=names[::-1] if c['schedule']=='reverse_sequential' else names
            for name in order:
                peer=next(n for n in names if n!=name);incoming=old if c['schedule']=='synchronous' else state
                own=1 if state[name]==1 else -1;other=1 if incoming[peer]==1 else -1
                margin=float(coefficients[(models[name],record['scenario']['id'])]@np.array([1,own,other,own*other]))
                state[name]=1 if margin>=0 else 2
                observed=next(r for r in rows if r['agent']==name and r['round']==t)
                results.append(dict(task=source.stem,round=t,agent=name,prediction=state[name],observed=observed['strategy'],predicted_margin=margin))
    write(path/'analysis/group_prediction.json',dict(method='Autonomous binary factorial response map; trained only on separately constructed histories; two-agent conditions',rows=results))
