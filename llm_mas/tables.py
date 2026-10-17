"""Derived tables and explicit evidence coverage."""
from .trajectories import read_conversation
import csv
import json
from pathlib import Path
import math


def flatten(value,prefix=''):
    result={}
    for key,item in value.items():
        name=f'{prefix}.{key}' if prefix else key
        if isinstance(item,dict):result.update(flatten(item,name))
        elif not isinstance(item,list):result[name]=item
    return result


def table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.with_suffix('.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    def escape(v):
        text=str(v if v is not None else '')
        for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:text=text.replace(a,b)
        return text
    # Compact tables only; detailed machine-readable tables stay CSV.
    if len(keys)<=8:
        lines=[r'\begin{tabular}{'+'l'*len(keys)+'}', ' & '.join(map(escape,keys))+r' \\ \hline']
        lines+=[' & '.join(escape(r.get(k,'')) for k in keys)+r' \\' for r in rows]
        lines += [r'\end{tabular}'];path.with_suffix('.tex').write_text('\n'.join(lines))


def export(path):
    import numpy as np
    folder=path/'tables';outcomes=[];changes=[]
    for file in sorted((path/'conversations').glob('*.json')):
        record=read_conversation(file);rows=record['rows'];states={}
        for r in rows:states.setdefault(r['round'],{})[r['agent']]=r['strategy']
        consensus=[t for t,s in states.items() if all(choice in (1,2) for choice in s.values()) and len(set(s.values()))==1]
        stable=next((t for t in consensus if all(len(set(states[k].values()))==1 and states[k]==states[t] for k in states if k>=t)),None)
        for agent in dict.fromkeys(r['agent'] for r in rows):
            rs=[r for r in rows if r['agent']==agent]
            outcomes.append(dict(task=file.stem,agent=agent,initial=rs[0]['strategy'],final=rs[-1]['strategy'],switches=sum(a['strategy'] in (1,2) and b['strategy'] in (1,2) and a['strategy']!=b['strategy'] for a,b in zip(rs,rs[1:])),stable_agreement_round=stable,final_margin=rs[-1]['post']['margin'],unique_messages=len({r['message'] for r in rs})))
            for stage in ('decision','post'):
                pairs=[(a,b) for a,b in zip(rs,rs[1:]) if a[stage] is not None and b[stage] is not None and 'sentiment' in a and 'sentiment' in b]
                x=np.array([abs(b[stage]['stance']-a[stage]['stance']) for a,b in pairs]);y=np.array([abs(b['sentiment']-a['sentiment']) for a,b in pairs])
                corr=float(np.corrcoef(x,y)[0,1]) if len(x)>1 and x.std()>0 and y.std()>0 else None
                changes.append(dict(task=file.stem,agent=agent,stage=stage,transitions=len(x),absolute_change_correlation=corr,both_changed=int(((x!=0)&(y!=0)).sum()),stance_only=int(((x!=0)&(y==0)).sum()),sentiment_only=int(((x==0)&(y!=0)).sum())))
        table(folder/(file.stem+'_transitions'),[dict(round=t,state=''.join(str(s[n]) for n in sorted(s))) for t,s in states.items()])
    table(folder/'outcomes',outcomes);table(folder/'stance_sentiment_changes',changes)
    for source in (path/'measurements').glob('*.json'):
        items=json.loads(source.read_text());rows=[]
        for item in items:
            base=flatten(item)
            if 'effects' in item:
                rows.extend(dict(base,**flatten(effect)) for effect in item['effects'])
            else:rows.append(base)
        table(folder/source.stem,rows)
    for source in (path/'analysis').glob('*.json'):
        data=json.loads(source.read_text())
        if isinstance(data,dict) and 'rows' in data:table(folder/source.stem,[flatten(r) for r in data['rows']])
