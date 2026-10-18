"""Static scientific figures and a linked HTML report for offline identification."""
from collections import defaultdict
import html
import json
from pathlib import Path
import numpy as np

LABELS={'persistence':'Persistence','control_drift':'Control drift','self_dynamics':'Self dynamics','coupled':'Coupled','configuration_coupled':'Configuration coupled','degroot':'DeGroot','friedkin_johnsen':'Friedkin–Johnsen'}


def short(composition):
    return '/'.join(model.split('/')[1].split('-')[0] for model in composition.split(' | '))


def generate(output,rows,predictions,fits,actual):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=output/'figures';folder.mkdir()
    summary=json.loads((output/'summary.json').read_text());configs=json.loads((output/'by_configuration.json').read_text())
    methods=list(LABELS)
    selected=[r for r in summary if r['subset']=='all_finite']
    fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    for col,schedule in enumerate(('synchronous','no_peer')):
        for row,(metric,label) in enumerate((('stance_mae','Stance MAE'),('margin_rmse','Margin RMSE'))):
            ax=axes[row,col];x=np.arange(len(methods))
            for offset,mode,color in ((-.18,'one_step','#0072B2'),(.18,'recursive','#D55E00')):
                values=[next(r[metric] for r in selected if r['schedule']==schedule and r['mode']==mode and r['method']==method) for method in methods]
                ax.bar(x+offset,values,width=.36,label=mode.replace('_',' '),color=color)
            ax.set_xticks(x,[LABELS[m] for m in methods],rotation=35,ha='right');ax.set_ylabel(label);ax.set_title(schedule.replace('_',' '));ax.legend()
    fig.suptitle('Held-out prediction errors · equal weighting of questions');fig.savefig(folder/'prediction_errors.png',dpi=170);plt.close(fig)

    compositions=sorted({r['composition'] for r in rows});questions=sorted({r['question'] for r in rows})
    byquestion=json.loads((output/'by_question.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    for ax,mode in zip(axes,('one_step','recursive')):
        matrix=[]
        for q in questions:
            vals=[]
            for comp in compositions:
                def error(method):
                    return next(r['stance_mae'] for r in byquestion if r['question']==q and r['composition']==comp and r['schedule']=='synchronous' and r['subset']=='all_finite' and r['mode']==mode and r['method']==method)
                vals.append(error('configuration_coupled')-error('self_dynamics'))
            matrix.append(vals)
        matrix=np.array(matrix);bound=max(np.max(abs(matrix)),.001)
        im=ax.imshow(matrix,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
        ax.set_xticks(range(len(compositions)),[short(c) for c in compositions],rotation=25,ha='right');ax.set_yticks(range(len(questions)),[q[:8] for q in questions]);ax.set_title(mode.replace('_',' '))
        for i in range(len(questions)):
            for j in range(len(compositions)):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=ax,label='Configuration coupled minus self MAE; negative is better')
    fig.suptitle('Same-question configuration comparisons · discussion');fig.savefig(folder/'question_configuration_errors.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,mode in zip(axes,('one_step','recursive')):
        for method in ('self_dynamics','coupled','configuration_coupled','control_drift'):
            rr=[r for r in json.loads((output/'by_round.json').read_text()) if r['subset']=='all_finite' and r['schedule']=='synchronous' and r['method']==method and r['mode']==mode]
            ax.plot([r['round'] for r in rr],[r['stance_mae'] for r in rr],marker='o',label=LABELS[method])
        ax.set_title(mode.replace('_',' '));ax.set_xlabel('Update round');ax.set_ylabel('Question-macro stance MAE');ax.legend(fontsize=8)
    fig.savefig(folder/'error_by_round.png',dpi=180);plt.close(fig)

    outcomes=json.loads((output/'configuration_outcomes.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for ax,metric,title in zip(axes,('explicit_match_strict','predicted_correct'),('Match actual final group outcome','Predicted correct unanimous answer')):
        x=np.arange(len(compositions));mm=('always_yes','self_dynamics','coupled','configuration_coupled','control_drift')
        for index,method in enumerate(mm):
            values=[next(r[metric] for r in outcomes if r['composition']==comp and r['level']=='group' and r['schedule']=='synchronous' and r['subset']=='all_finite' and r['method']==method and r['mode']!='one_step') for comp in compositions]
            ax.bar(x+(index-2)*.15,values,width=.15,label=LABELS.get(method,'Always Yes'))
        ax.set_xticks(x,[short(c) for c in compositions]);ax.set_ylim(0,1);ax.set_ylabel('Question-macro rate');ax.set_title(title);ax.legend(fontsize=8)
    fig.savefig(folder/'final_outcomes.png',dpi=180);plt.close(fig)

    # Plot raw, unclipped recursive margin growth separately from bounded stance.
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for method in methods:
        rr=[r for r in predictions if r['subset']=='all_finite' and r['schedule']=='synchronous' and r['mode']=='recursive' and r['method']==method]
        rounds=sorted({r['round'] for r in rr})
        ax.plot(rounds,[max(abs(r['predicted_margin']) for r in rr if r['round']==turn) for turn in rounds],marker='o',label=LABELS[method])
    ax.set_yscale('symlog',linthresh=1);ax.set_xlabel('Update round');ax.set_ylabel('Maximum absolute predicted margin');ax.set_title('Recursive magnitude diagnostic · no margin clipping');ax.legend(fontsize=8,ncol=2)
    fig.savefig(folder/'recursive_magnitude.png',dpi=180);plt.close(fig)

    colors={'Agent A':'#0072B2','Agent B':'#D55E00'}
    question_text={}
    source=Path(json.loads((output/'manifest.json').read_text())['source'])
    for path in sorted((source/'conversations').glob('*.json')):
        record=json.loads(path.read_text());question_text[record['question_id']]=record['scenario']['question']
    for question in questions:
        fig,axes=plt.subplots(len(compositions),2,figsize=(14,12),layout='constrained',sharex=True,sharey=True)
        for i,comp in enumerate(compositions):
            for j,schedule in enumerate(('synchronous','no_peer')):
                ax=axes[i,j]
                selected_rows=[r for r in rows if r['question']==question and r['composition']==comp and r['schedule']==schedule]
                rounds=sorted({r['round'] for r in selected_rows})
                for agent in sorted({r['agent'] for r in selected_rows}):
                    own_rows=[r for r in selected_rows if r['agent']==agent];color=colors.get(agent,'#009E73')
                    starts=[float(np.tanh(r['own']/2)) for r in own_rows if r['round']==1]
                    for conversation in sorted({r['conversation'] for r in own_rows}):
                        trajectory=sorted([r for r in own_rows if r['conversation']==conversation],key=lambda r:r['round'])
                        ax.plot([0]+[r['round'] for r in trajectory],[np.tanh(trajectory[0]['own']/2)]+[np.tanh(r['target']/2) for r in trajectory],color=color,alpha=.18,lw=1)
                    observed=[np.mean([np.tanh(r['target']/2) for r in own_rows if r['round']==turn]) for turn in rounds]
                    ax.plot([0]+rounds,[np.mean(starts)]+observed,color=color,lw=2,label=f'{agent}: actual')
                    for method,style in (('configuration_coupled','--'),('self_dynamics',':')):
                        pp=[r for r in predictions if r['question']==question and r['composition']==comp and r['schedule']==schedule and r['agent']==agent and r['subset']=='all_finite' and r['mode']=='recursive' and r['method']==method]
                        ax.plot([0]+rounds,[np.mean(starts)]+[np.mean([r['prediction'] for r in pp if r['round']==turn]) for turn in rounds],color=color,ls=style,lw=1.6,label=f'{agent}: {LABELS[method]}')
                ax.axhline(0,color='gray',lw=.6);ax.set_ylim(-1.05,1.05);ax.set_title(f'{short(comp)} · {schedule.replace("_"," ")}');ax.set_ylabel('Mean stance');ax.set_xlabel('Round');ax.legend(fontsize=6,ncol=2)
        fig.suptitle(f'{question_text[question]}\nHeld-out recursive predictions; means of 3 repetitions; faint lines = actual repetitions',fontsize=11)
        fig.savefig(folder/f'trajectory_{question}.png',dpi=160);plt.close(fig)

    text=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Configuration-aware system identification</title>',
          '<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17212b}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{padding:.5rem;border:1px solid #aaa}a{color:#005ea8}</style>',
          '<h1>Configuration-aware system identification</h1>',
          '<p>Exploratory nested question-held-out evaluation of existing conversations. No language models were run. Predictions begin from observed initialization. Coefficients describe output dynamics, not psychological properties or causal influence.</p>',
          '<p>Primary fitting target: raw Yes/No likelihood margin. Stance is tanh(margin/2). Ridge tuning uses only inner training questions and minimizes one-step margin MSE. Recursive forecasts use predicted states and fixed delivery timing, without future observed states or text. All-finite results are shown; CSV/JSON also include valid-response sensitivity.</p>',
          '<p>The self model includes own state, initial-minus-own state, round, slot, and a delivery-presence indicator. Coupled adds delivered peer-minus-own state. Configuration coupled adds regularized composition-and-slot adjustments. Classical models are refitted on margins. Cross-checkpoint score scales are uncalibrated.</p>',
          '<h2>Held-out discussion errors</h2><table><tr><th>Method</th><th>One-step stance MAE</th><th>Recursive stance MAE</th></tr>']
    for method in methods:
        vals=[next(r['stance_mae'] for r in selected if r['schedule']=='synchronous' and r['mode']==mode and r['method']==method) for mode in ('one_step','recursive')]
        text.append(f'<tr><td>{LABELS[method]}</td><td>{vals[0]:.4f}</td><td>{vals[1]:.4f}</td></tr>')
    text.append('</table>')
    for filename,title in [('prediction_errors','Prediction errors'),('question_configuration_errors','Question and configuration differences'),('error_by_round','Error accumulation'),('final_outcomes','Final outcomes'),('recursive_magnitude','Recursive magnitude')]:
        text.append(f'<h2>{title}</h2><img src="figures/{filename}.png" alt="{title}">')
    text.append('<h2>Same-question trajectories across configurations</h2><p>Solid lines show observed means, dashed lines configuration-aware forecasts, dotted lines self-dynamics forecasts. Both predicted trajectories start from observed round zero. Repetition variability is shown by faint observed trajectories.</p>')
    for question in questions:
        text.append(f'<h3>{html.escape(question_text[question])}</h3><img src="figures/trajectory_{question}.png" alt="Actual and predicted trajectories for question {question}, across four compositions and two conditions">')
    text.append('<h2>Exports</h2><ul>')
    for filename in sorted(output.glob('*.csv')):text.append(f'<li><a href="{filename.name}">{filename.name}</a></li>')
    text.append('</ul><p>Inspect residual diagnostics and fold parameter stability alongside prediction errors. There are only ten independent question clusters; do not interpret repetitions as independent question evidence.</p></html>')
    (output/'index.html').write_text('\n'.join(text))
