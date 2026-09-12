"""Match exported primary outcomes with fixed-horizon and reversed-presentation runs."""
import argparse
import csv
from pathlib import Path
from .full_study_analysis import table

def main():
    p=argparse.ArgumentParser();p.add_argument('--primary',type=Path,required=True);p.add_argument('--fixed',type=Path,required=True);p.add_argument('--reversed',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    def read(path):return list(csv.DictReader((path/'tables/outcomes.csv').open()))
    def key(r):return tuple(r[k] for k in ['scenario','models','persona','initialization','schedule','decision_mode'])
    original={key(r):r for r in read(a.primary)};rows=[]
    for label,path in [('fixed_horizon',a.fixed),('reversed_presentation',a.reversed)]:
        for r in read(path):
            b=original.get(key(r))
            if b is None:raise ValueError('Missing primary match')
            rows.append(dict(comparison=label,primary_task=b['task'],other_task=r['task'],scenario=r['scenario'],decision_mode=r['decision_mode'],
                schedule=r['schedule'],primary_outcome=b['outcome'],other_outcome=r['outcome'],outcome_changed=b['outcome']!=r['outcome'],
                primary_round=int(b['final_round']),other_round=int(r['final_round']),later_revision=r['later_revision'],potential_rounds_saved=r['potential_rounds_saved'],potential_tokens_saved=r.get('potential_tokens_saved','')))
    a.output.mkdir(parents=True,exist_ok=True);table(a.output/'matched_comparisons.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    labels=['fixed_horizon','reversed_presentation']
    axes[0].bar(labels,[sum(r['outcome_changed'] for r in rows if r['comparison']==k) for k in labels]);axes[0].set_ylabel('Matched final outcomes changed')
    fixed=[r for r in rows if r['comparison']=='fixed_horizon' and r['potential_rounds_saved']!='']
    axes[1].hist([int(r['potential_rounds_saved']) for r in fixed]);axes[1].set(xlabel='Potential rounds saved',ylabel='Conversations with stopping trigger')
    for ext in ['png','pdf']:fig.savefig(a.output/f'comparisons.{ext}',bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':main()
