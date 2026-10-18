"""Figures, question pages and excerpt-linked HTML for descriptive deliberation analysis."""
from collections import Counter,defaultdict
import html
import json
from pathlib import Path
import statistics

import numpy as np

from model_saved_pilot import export
from report_question_deliberation import COMPOSITIONS,STATES,mean,link,coding_criteria,valid,endpoint_status,endpoint_transitions

STYLE='''<style>body{font:16px/1.5 system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#18232e}img{max-width:100%;height:auto}table{border-collapse:collapse;margin:1rem 0;font-size:14px}th,td{padding:.4rem .6rem;border:1px solid #adb5bd;text-align:left;vertical-align:top}th{background:#eef3f8}a{color:#005ea8}pre,blockquote{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:1rem}details{border:1px solid #b7c2cc;padding:.6rem;margin:.6rem 0}summary{cursor:pointer;font-weight:600}.scroll{overflow-x:auto}.note{border-left:4px solid #496985;padding:.5rem 1rem}section{scroll-margin-top:1rem}small{color:#465361}</style>'''


def e(value):return html.escape(str(value))


def page(title):return [f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{e(title)}</title>',STYLE,f'<h1>{e(title)}</h1>']


def table(rows,columns):
    text=['<div class="scroll"><table><thead><tr>'+''.join(f'<th scope="col">{e(label)}</th>' for key,label in columns)+'</tr></thead><tbody>']
    for row in rows:
        values=[]
        for key,label in columns:
            value=row.get(key)
            if value is None:value='—'
            elif isinstance(value,float):value=f'{value:.3f}'
            values.append(f'<td>{e(value)}</td>')
        text.append('<tr>'+''.join(values)+'</tr>')
    text.append('</tbody></table></div>');return '\n'.join(text)


def fig_tag(filename,title,caption,prefix=''):
    return f'<h2>{e(title)}</h2><figure><img src="{prefix}figures/{filename}.png" alt="{e(title)}"><figcaption>{e(caption)}</figcaption></figure>'


def generate(output,records,turns,trajectories,groups,conversations,deliveries,pairs,configuration,question,examples,benchmarks,failures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    figures=output/'figures';figures.mkdir();pages=output/'questions';pages.mkdir()
    colors={'Yes':'#0072B2','No':'#D55E00','Split':'#CC79A7','Unknown':'#999999'}
    conditions=('synchronous','no_peer');condition_names={'synchronous':'Discussion','no_peer':'No peer'}
    qids=sorted({d['question_id'] for d in records});questions={d['question_id']:d['scenario']['question'] for d in records}
    labels={d['question_id']:d['evaluation']['reference_answer'] for d in records}
    recurrence=[]
    for comp in COMPOSITIONS:
        rr=[r for r in pairs if r['composition']==comp]
        row=dict(composition=comp,questions=len(qids),paired_blocks=len(rr))
        for name,key in [('consensus','consensus_difference'),('correctness','final_correct_difference'),('reversals','reversals_difference'),('stance_movement','absolute_stance_change_difference'),('self_repetition','argument_repeat_rate_difference')]:
            effects=[mean([r[key] for r in rr if r['question_id']==q]) for q in qids]
            row[name+'_higher_questions']=sum(v>1e-10 for v in effects);row[name+'_lower_questions']=sum(v< -1e-10 for v in effects);row[name+'_equal_questions']=sum(abs(v)<=1e-10 for v in effects)
        recurrence.append(row)
    export(output/'tables','question_recurrence',recurrence)

    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    metrics=[('final_consensus_rate','Final consensus / all conversations'),('reversal_rate','Reversals / eligible adjacent turns'),('mean_absolute_stance_update','Mean absolute stance update'),('invalid_response_rate','Invalid / recorded responses (includes shared starts)')]
    for ax,(metric,title) in zip(axes.flat,metrics):
        x=np.arange(4)
        for offset,condition,color in ((-.18,'synchronous','#0072B2'),(.18,'no_peer','#D55E00')):
            values=[next(r[metric] for r in configuration if r['composition']==comp and r['schedule']==condition) for comp in COMPOSITIONS]
            ax.bar(x+offset,values,width=.36,label=condition_names[condition],color=color)
        ax.set_xticks(x,COMPOSITIONS);ax.set_title(title,fontsize=11);ax.legend()
    fig.suptitle('Deliberation patterns across configurations');fig.savefig(figures/'configuration_patterns.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for col,condition in enumerate(conditions):
        x=np.arange(4)
        for offset,metric,coverage,title,color in ((-.18,'initial_accuracy','initial_valid_answers','Initial','#009E73'),(.18,'final_accuracy','final_valid_answers','Final','#0072B2')):
            values=[next(r[metric] for r in configuration if r['composition']==comp and r['schedule']==condition) for comp in COMPOSITIONS]
            coverage_values=[next(r[coverage]/r['agent_trajectories'] for r in configuration if r['composition']==comp and r['schedule']==condition) for comp in COMPOSITIONS]
            axes[0,col].bar(x+offset,values,width=.36,label=title,color=color)
            bars=axes[1,col].bar(x+offset,coverage_values,width=.36,label=title,color=color)
            for bar,comp in zip(bars,COMPOSITIONS):
                row=next(r for r in configuration if r['composition']==comp and r['schedule']==condition)
                axes[1,col].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.015,f"{row[coverage]}/{row['agent_trajectories']}",ha='center',fontsize=8)
        for ax in axes[:,col]:ax.set_xticks(x,COMPOSITIONS);ax.set_ylim(0,1.1)
        axes[0,col].legend()
        axes[0,col].set_title(condition_names[condition]);axes[0,col].set_ylabel('Correct answers / valid answers')
        axes[1,col].set_ylabel('Valid answers / recorded endpoints')
    fig.suptitle('Initial and final correctness with answer coverage shown separately');fig.savefig(figures/'initial_final_correctness.png',dpi=180);plt.close(fig)

    endpoint_rows=endpoint_transitions(trajectories,('composition','schedule'))
    statuses=('Correct','Incorrect','Invalid')
    if any(r['initial_outcome']=='Unlabeled' or r['final_outcome']=='Unlabeled' for r in endpoint_rows):statuses+=('Unlabeled',)
    maximum=max(r['count'] for r in endpoint_rows) or 1
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for i,condition in enumerate(conditions):
        for j,comp in enumerate(COMPOSITIONS):
            ax=axes[i,j];rr=[r for r in endpoint_rows if r['composition']==comp and r['schedule']==condition]
            matrix=np.array([[next((r['count'] for r in rr if r['initial_outcome']==initial and r['final_outcome']==final),0) for final in statuses] for initial in statuses])
            total=sum(r['count'] for r in rr);im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=maximum)
            ax.set_xticks(range(len(statuses)),statuses,rotation=25,ha='right');ax.set_yticks(range(len(statuses)),statuses)
            ax.set_xlabel('Final outcome');ax.set_ylabel('Initial outcome');ax.set_title(f'{comp} · {condition_names[condition]} · N={total}')
            for ii in range(len(statuses)):
                for jj in range(len(statuses)):
                    n=matrix[ii,jj];ax.text(jj,ii,f'{n}\n{100*n/total:.1f}%',ha='center',va='center',color='white' if n>maximum*.55 else 'black',fontsize=10)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='Number of agent trajectories')
    fig.suptitle('Initial-to-final answer transitions · invalid responses are a separate outcome\nCell percentages use all endpoint pairs; this does not summarize intermediate reversals')
    fig.savefig(figures/'initial_final_transitions.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax,initial in zip(axes,(True,False)):
        column='initial_state' if initial else 'final_state';x=np.arange(8);bottom=np.zeros(8)
        for groupstate in STATES:
            values=[mean([r[column]==groupstate for r in conversations if r['composition']==comp and r['schedule']==condition]) for comp in COMPOSITIONS for condition in conditions]
            ax.bar(x,values,bottom=bottom,label=groupstate,color=colors[groupstate]);bottom+=np.array(values)
        ax.set_xticks(x,[f'{c}\n{"D" if s=="synchronous" else "C"}' for c in COMPOSITIONS for s in conditions]);ax.set_ylim(0,1);ax.set_title('Initialization' if initial else 'Final recorded round');ax.set_ylabel('Proportion of conversations');ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Group outcomes · D: discussion; C: matched control');fig.savefig(figures/'group_states.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained')
    for ax,key,title in zip(axes,('consensus_difference','reversals_difference','argument_repeat_rate_difference'),('Final consensus difference','Reversal count difference per agent','Exact self-repetition rate difference')):
        matrix=np.array([[mean([r[key] for r in pairs if r['question_id']==q and r['composition']==comp]) for comp in COMPOSITIONS] for q in qids]);limit=max(np.max(abs(matrix)),.001)
        im=ax.imshow(matrix,vmin=-limit,vmax=limit,cmap='RdBu_r',aspect='auto');ax.set_xticks(range(4),COMPOSITIONS);ax.set_yticks(range(len(qids)),[q[:8] for q in qids]);ax.set_title(title,fontsize=10)
        for i in range(len(qids)):
            for j in range(4):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Same-question matched differences: discussion minus no peer; signs do not denote benefit');fig.savefig(figures/'question_matched_differences.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,metrics,title in zip(axes,[('exact_self_repeat_rate','exact_peer_reuse_rate'),('reference_cue_rate','agreement_cue_rate','challenge_cue_rate')],['Exact normalized explanation reuse','Explicit surface cues; not semantic engagement labels']):
        for i,metric in enumerate(metrics):
            values=[next(r[metric] for r in configuration if r['composition']==comp and r['schedule']=='synchronous') or 0 for comp in COMPOSITIONS]
            width=.7/len(metrics);ax.bar(np.arange(4)+(i-(len(metrics)-1)/2)*width,values,width=width,label=metric.replace('_rate','').replace('_',' '))
        ax.set_xticks(range(4),COMPOSITIONS);ax.set_ylabel('Proportion of discussion updates/deliveries');ax.set_title(title,fontsize=10);ax.legend(fontsize=8)
    fig.savefig(figures/'textual_features.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,condition in zip(axes,conditions):
        for comp in COMPOSITIONS:
            rr=[r for r in groups if r['composition']==comp and r['schedule']==condition];rounds=sorted({r['round'] for r in rr})
            ax.plot(rounds,[mean([r['consensus'] for r in rr if r['round']==turn]) for turn in rounds],marker='o',label=comp)
        ax.set_ylim(0,1);ax.set_xlabel('Round');ax.set_ylabel('Consensus / all conversations');ax.set_title(condition_names[condition]);ax.legend()
    fig.savefig(figures/'consensus_over_rounds.png',dpi=180);plt.close(fig)

    if benchmarks:
        fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
        methods=('persistence','control_drift','self_dynamics','coupled','configuration_coupled','degroot','friedkin_johnsen')
        for ax,mode in zip(axes,('one_step','recursive')):
            values=[next(r['stance_mae'] for r in benchmarks if r['method']==method and r['mode']==mode and r['subset']=='all_finite' and r['schedule']=='synchronous') for method in methods]
            ax.bar(range(len(methods)),values,color='#0072B2');ax.set_xticks(range(len(methods)),[m.replace('_',' ') for m in methods],rotation=35,ha='right');ax.set_ylabel('Held-out question macro stance MAE');ax.set_title(mode.replace('_',' '))
        fig.suptitle('Limited numerical benchmarks · explanatory boundary, not the primary result');fig.savefig(figures/'limited_benchmarks.png',dpi=180);plt.close(fig)

    plot_index=[]
    for q in qids:
        repetitions=sorted({r['repetition'] for r in turns if r['question_id']==q})
        for repetition in repetitions:
            fig,axes=plt.subplots(4,3,figsize=(17,12),layout='constrained')
            for i,comp in enumerate(COMPOSITIONS):
                matched=matched_plot_data(turns,groups,q,comp,repetition)
                index=dict(question_id=q,composition=comp,repetition=repetition,figure=f'figures/question_{q}_rep_{repetition}.png')
                for j,condition in enumerate(conditions):
                    ax=axes[i,j];rr=matched[condition]['turns'];conversation=matched[condition]['conversation'];index[condition+'_conversation']=conversation
                    for agent,color in (('Agent A','#0072B2'),('Agent B','#D55E00')):
                        own=sorted([r for r in rr if r['agent']==agent],key=lambda r:r['round'])
                        ax.plot([r['round'] for r in own],[r['stance'] for r in own],color=color,lw=2,marker='o',markersize=3,label=agent)
                    ax.set_ylim(-1.05,1.05);ax.axhline(0,color='gray',lw=.6);ax.set_title(f'{comp} · {condition_names[condition]} · {conversation}',fontsize=10);ax.set_xlabel('Round');ax.set_ylabel('Measured stance');ax.legend(fontsize=7)
                plot_index.append(index)
                ax=axes[i,2];rounds=matched['rounds']
                matrix=np.array([[STATES.index(r['state']) for r in matched[condition]['groups']] for condition in conditions])
                cmap=ListedColormap([colors[state] for state in STATES]);ax.imshow(matrix,cmap=cmap,norm=BoundaryNorm(np.arange(-.5,4.5),4),aspect='auto')
                ax.set_yticks([0,1],['Discussion','No peer'],fontsize=9);ax.set_xticks(range(len(rounds)),rounds);ax.set_xlabel('Round');ax.set_title(f'{comp} · same-run explicit-answer group states',fontsize=10)
                for ii in range(2):
                    for jj in range(len(rounds)):
                        text=('Yes','No','Split','Invalid')[matrix[ii,jj]]
                        ax.text(jj,ii,text,ha='center',va='center',color='black' if text=='Split' else 'white',fontsize=8)
            fig.suptitle(f'{questions[q]}\nRun {repetition+1} (saved repetition {repetition}) · exact matched discussion/control trajectories; no averaging · dataset label: {labels[q]}',fontsize=11)
            fig.savefig(figures/f'question_{q}_rep_{repetition}.png',dpi=160);plt.close(fig)
        question_page(output,q,questions[q],labels[q],records,question,examples,failures)
    export(output/'tables','question_plot_index',plot_index)

    report=page('Question-centered deliberation report')
    report+=['<p class="note">Existing pilot only: 10 questions, 4 ordered checkpoint compositions, 3 repetitions, synchronous discussion and matched no-peer controls. The 120 pairs share exactly identical initialization records. No messages were regenerated or repaired. Q = Qwen; S = SmolLM2. Composition order identifies agent slots, not speaking order.</p>',
        '<p>The primary outcomes are output evolution, disagreement, convergence and recorded message relationships. Correctness is a secondary evaluator-only outcome. Ten questions are the independent question units; turns and repetitions do not create additional independent questions.</p>',
        '<nav><a href="#questions">Question comparisons</a> · <a href="#patterns">Patterns</a> · <a href="#coding">Message interpretation</a> · <a href="#modeling">Limited modeling</a> · <a href="#exports">Tables</a></nav>',
        '<h2 id="questions">Question-centered comparisons</h2><p>Each question page contains all eight configuration/condition combinations, matched effects, selectable single-repetition trajectories with aligned group states, and expandable transcripts with the exact messages delivered before every response.</p><ul>']
    for q in qids:report.append(f'<li><a href="questions/{q}.html">{e(questions[q])}</a> <small>Label: {labels[q]}; {q}</small></li>')
    report+=['</ul><h2 id="patterns">Configuration synthesis</h2>',table(configuration,[('composition','Composition'),('schedule','Condition'),('initial_disagreement_rate','Initial disagreement / all'),('final_consensus_rate','Final consensus / all'),('reversal_rate','Reversals / eligible turns'),('persistent_answer_rate','Persistence / fully valid trajectories'),('repeated_switching_rate','Repeated switching / all trajectories'),('maintained_disagreement_rate','Always split / all conversations'),('invalid_response_rate','Invalid / recorded responses')]),
        '<p>Persistence requires valid unchanged answers at every round. Repeated switching counts at least two adjacent-valid reversals; invalid rounds are not bridged. Maintained disagreement requires valid disagreement at all recorded rounds. Consensus can be transient; first and durable consensus times are in the conversation table.</p>',
        fig_tag('configuration_patterns','Behavioral patterns','Thirty conversations per composition and condition. Invalid-response rates include duplicated shared initialization records; this is a recorded-response rate, not a unique-generation failure rate.'),
        fig_tag('group_states','Initial and final group states','Unknown means at least one invalid answer. Initial state distributions must match across paired branches.'),
        fig_tag('consensus_over_rounds','Consensus over rounds','Consensus means two valid identical explicit answers; agreement is not correctness.'),
        fig_tag('question_matched_differences','Same-question matched comparisons','Each cell averages three matched repetitions. Positive differences indicate more of a pattern under discussion, not necessarily improvement.'),
        '<h2>Recurrence across questions</h2>',table(recurrence,[('composition','Composition'),('consensus_higher_questions','Higher consensus'),('consensus_lower_questions','Lower consensus'),('consensus_equal_questions','Equal consensus'),('reversals_higher_questions','More reversals'),('reversals_lower_questions','Fewer reversals'),('stance_movement_higher_questions','More stance movement')]),
        '<p>Counts are out of ten questions, comparing discussion with its matched control within each question and composition. Pooled proportions describe this balanced pilot. Conditional rates can have different valid denominators; consult exported eligible counts.</p>',
        fig_tag('initial_final_correctness','Correctness in context','Correctness is the fraction correct among valid answers at that endpoint. The lower panels show valid-answer coverage with explicit counts. Initial and final valid subsets may differ; use the transition matrix to track the same agents.'),
        fig_tag('initial_final_transitions','Initial-to-final correctness transitions','Rows show the initial outcome and columns the final outcome for the same agent. Correct→incorrect and incorrect→correct are distinct cells. Invalid endpoints are retained in their own row/column, never counted as incorrect. Percentages use all trajectories; intermediate behavior may include additional reversals.'),
        '<h2 id="coding">Message-linked interpretation and coding criteria</h2>',table(coding_criteria(),[('code','Code'),('definition','Criterion'),('status','Interpretation boundary')]),
        '<p>Automatic cue counts below are surface features. They are not frequencies of semantic adoption, challenge, or ignoring. All delivered events have an editable review template. No observable uptake is distinct from claiming an internal process of ignoring.</p>',
        fig_tag('textual_features','Observable textual features','Exact reuse uses normalized explanations; existing adjacent cosine scores are retained in tables. Similar wording does not establish agreement, correctness, or causal copying.'),
        '<h2>Illustrative transcript interpretations</h2><p>Thirteen assistant-inspected cases were selected purposively to illustrate different patterns and failure cases. They are not a representative coded sample, and no independent human annotation or inter-rater reliability is claimed. Statements in excerpts are agent outputs, not externally verified facts.</p>']
    for example in examples:report.append(excerpt(example))
    report+=['<h2 id="modeling">Limited numerical-model interpretation</h2>','<p>The numerical benchmarks ask how much recorded state movement can be approximated without interpreting argument content. They cannot identify the meaning of an argument or a causal influence mechanism. They remain a secondary boundary check, not the main research conclusion.</p>']
    if benchmarks:
        report+=[fig_tag('limited_benchmarks','What reduced models capture','Fits and predictions are reused unchanged from the separate system-identification analysis; models were not retrained for this report.'),
            '<p>Coupling improves aggregate numerical prediction relative to self dynamics, but configuration adjustments do not consistently improve it. The coupled recursive model matches 67.5% of final discussion group outcomes versus 70% for always unanimous Yes, and predicts consensus in 115/120 cases versus 93/120 observed. These pilot findings show that a smooth reduced model misses much of the remaining disagreement and switching.</p>',
            '<p>Largest one-step forecast errors are linked below to the actual delivery context. They identify cases needing explanation; high forecast error does not by itself establish a linguistic cause.</p><ul>']
        for r in failures[:10]:report.append(f'<li><a href="{e(r["transcript"])}">{e(r["composition"])} · {e(r["conversation"])} · {e(r["response_id"])}</a>: absolute stance error {r["absolute_error"]:.3f}</li>')
        report.append('</ul>')
    else:report.append('<p>No matching saved modeling analysis was supplied; no new models were fitted.</p>')
    report+=['<h2 id="exports">Exported tables and provenance</h2><ul>']
    for file in sorted((output/'tables').glob('*.csv')):report.append(f'<li><a href="tables/{file.name}">{file.name}</a> · <a href="tables/{file.stem}.json">JSON</a></li>')
    report+=['</ul><p><a href="manifest.json">Input hashes, counts, definitions and limitations</a>. Shared initialization is duplicated in saved branch records. Native stance scales are not calibrated across checkpoints. Similarity and movement alone do not establish causal influence.</p></html>']
    (output/'index.html').write_text('\n'.join(report))


def excerpt(row,prefix=''):
    return f'''<details><summary>{e(row['composition'])} · {e(row['conversation'])} · {e(row['response_id'])}: {e(row['reviewed_interpretation'])}</summary>
<p>{e(row['interpretation_note'])}</p><p><a href="{prefix}{e(row['transcript'])}">Full response and delivery context</a></p>
<strong>Previous own message</strong><blockquote>{e(row['prior_message'])}</blockquote>
<strong>Delivered peer message ({e(row['source_id'])})</strong><blockquote>{e(row['delivered_message'])}</blockquote>
<strong>Next response</strong><blockquote>{e(row['response_message'])}</blockquote><small>{e(row['review_status'])}</small></details>'''


def matched_plot_data(turns,groups,question,composition,repetition):
    result={}
    for condition in ('synchronous','no_peer'):
        selected=[r for r in turns if r['question_id']==question and r['composition']==composition and r['repetition']==repetition and r['schedule']==condition]
        conversations={r['conversation'] for r in selected}
        if len(conversations)!=1:raise ValueError('A question panel must contain exactly one conversation')
        conversation=next(iter(conversations))
        rr=sorted([r for r in groups if r['conversation']==conversation],key=lambda r:r['round'])
        rounds=sorted({r['round'] for r in selected})
        if [r['round'] for r in rr]!=rounds:raise ValueError('Stance/group-state round mismatch')
        if any(sum(r['round']==turn for r in selected)!=2 for turn in rounds):raise ValueError('Missing agent measurement')
        for group in rr:
            answers=[r['answer'] for r in selected if r['round']==group['round']]
            expected='Unknown' if any(a not in ('Yes','No') for a in answers) else answers[0] if len(set(answers))==1 else 'Split'
            if group['state']!=expected:raise ValueError('Categorical group state does not match the plotted conversation')
        result[condition]=dict(conversation=conversation,turns=selected,groups=rr,rounds=rounds)
    left=result['synchronous'];right=result['no_peer']
    if left['rounds']!=right['rounds']:raise ValueError('Matched branches have different rounds')
    pair_ids={r['pair_id'] for c in (left,right) for r in c['turns']}
    if len(pair_ids)!=1 or None in pair_ids:raise ValueError('Discussion/control panel is not a matched pair')
    result['rounds']=left['rounds']
    return result


def repetition_selector(q,records):
    repetitions=sorted({d.get('repetition') for d in records if d['question_id']==q})
    text=['<h2>Matched single-run trajectories</h2><label for="repetition-select">Select repetition: </label><select id="repetition-select" onchange="document.querySelectorAll(\'[data-repetition]\').forEach(p=>p.hidden=p.dataset.repetition!==this.value)">']
    for repetition in repetitions:text.append(f'<option value="{repetition}">Run {repetition+1} (saved repetition {repetition})</option>')
    text.append('</select><p>All three columns use the selected repetition and identical round indices. Each composition shows one discussion conversation and its matched no-peer branch. Lines are exact recorded scores, without averaging. Group states use explicit answers, so a score-sign/answer mismatch may still occur.</p>')
    for repetition in repetitions:
        hidden='' if repetition==repetitions[0] else ' hidden'
        text.append(f'<div data-repetition="{repetition}"{hidden}>'+fig_tag(f'question_{q}_rep_{repetition}',f'Run {repetition+1}: matched discussion and no-peer trajectories','Source conversation IDs are printed on each stance panel. Yes/No means unanimous explicit answers; Split means disagreement; Invalid means at least one invalid answer. Native stance scales are uncalibrated across checkpoints.',prefix='../')+'</div>')
    return '\n'.join(text)


def question_page(output,q,text,label,records,question,examples,failures):
    selected=[d for d in records if d['question_id']==q]
    content=page(text);content+=['<p><a href="../index.html">Back to synthesis</a></p>',f'<p>Question ID: {q}; evaluator-only label: {label}. Each configuration has three matched repetitions. Invalid answers are kept visible.</p>',
        repetition_selector(q,records),
        '<h2>Initial states, updates and outcomes</h2>',table([r for r in question if r['question_id']==q],[('composition','Composition'),('schedule','Condition'),('initial_accuracy','Initial correct / valid'),('initial_valid_answers','Initial valid n'),('final_accuracy','Final correct / valid'),('final_valid_answers','Final valid n'),('initial_disagreement_rate','Initial split / all'),('final_consensus_rate','Final consensus / all'),('total_reversals','Reversals'),('aba_oscillations','ABA events'),('mean_absolute_stance_update','Mean |stance update|'),('exact_self_repeat_rate','Exact self repeat'),('initial_invalid_groups','Initial invalid groups'),('final_unknown_groups','Final invalid groups')]),
        '<p>Correctness proportions use valid answers only, with valid counts shown separately. Initial and final denominators may differ. Initial accuracy and categorical states match within each discussion/control pair. Across compositions, initial states can differ; final differences cannot be attributed solely to interaction.</p>']
    initial_final=json.loads((output/'tables'/'question_initial_final_transitions.json').read_text())
    content+=['<h2>Initial-to-final endpoint transitions</h2>',table([r for r in initial_final if r['question_id']==q and r['count']>0],[('composition','Composition'),('schedule','Condition'),('initial_outcome','Initial outcome'),('final_outcome','Final outcome'),('count','Agents'),('total_trajectories','All endpoint pairs')]),'<p>Invalid endpoints are separate from incorrect answers. These are endpoint comparisons; an agent may switch in intervening rounds and return to its original answer.</p>']
    for comp in COMPOSITIONS:
        a=next(r for r in question if r['question_id']==q and r['composition']==comp and r['schedule']=='synchronous');b=next(r for r in question if r['question_id']==q and r['composition']==comp and r['schedule']=='no_peer')
        content.append(f'<p><strong>{comp}:</strong> discussion versus control has {a["total_reversals"]} versus {b["total_reversals"]} valid adjacent reversals; final consensus in {round(a["final_consensus_rate"]*3)}/3 versus {round(b["final_consensus_rate"]*3)}/3 conversations; mean absolute stance updates {a["mean_absolute_stance_update"]:.3f} versus {b["mean_absolute_stance_update"]:.3f}. These describe the recorded trajectories rather than an inferred internal mechanism.</p>')
    content.append('<h2>Illustrative message interpretations</h2>')
    ee=[r for r in examples if r['question_id']==q]
    if not ee:content.append('<p>No purposively reviewed illustration for this question. Full delivery context and unreviewed coding candidates remain available below and in the exported review template.</p>')
    for row in ee:content.append(excerpt(row,prefix='../'))
    content.append('<h2>Transcripts and exact delivery timing</h2><p>Open a run to inspect all recorded responses. Source links point to the message actually delivered. Invalid public messages are preserved, with raw generated output available separately.</p>')
    for d in sorted(selected,key=lambda d:(COMPOSITIONS.index(d['_composition']),d.get('repetition',0),d['config']['schedule'])):
        conv=d['_conversation'];byid={r['response_id']:r for r in d['rows']}
        content.append(f'<details id="{conv}"><summary>{d["_composition"]} · {d["config"]["schedule"]} · repetition {d.get("repetition")} · {conv}</summary>')
        for r in d['rows']:
            content.append(f'<section id="{link(conv,r["response_id"])}"><h3>Round {r["round"]}: {e(r["agent"])} · {e(r["model"].split("/")[1])}</h3><p>Answer: {e(r["answer"])}; status: {e(r["choice_status"])}; margin: {r["post"]["margin"]:.3f}; stance: {r["post"]["stance"]:.3f}</p>')
            if r.get('deliveries'):
                for delivery in r['deliveries']:
                    content.append(f'<p><strong>Delivered before this response:</strong> <a href="#{link(conv,delivery["source_id"])}">{e(delivery["speaker"])} · source round {delivery["source_round"]} · {e(delivery["source_id"])}</a></p><blockquote>{e(delivery["message"])}</blockquote>')
            else:content.append('<p>No peer message delivered before this response.</p>')
            content.append(f'<strong>Public response</strong><blockquote>{e(r["message"])}</blockquote>')
            if not valid(r):content.append(f'<details><summary>Raw generated output for invalid response</summary><pre>{e(r.get("generation",{}).get("output",""))}</pre></details>')
            content.append('</section>')
        content.append('</details>')
    content+=['<script>function reveal(){const id=decodeURIComponent(location.hash.slice(1));const target=document.getElementById(id);if(target){for(let p=target.parentElement;p;p=p.parentElement){if(p.tagName==="DETAILS")p.open=true;}target.scrollIntoView();}}addEventListener("hashchange",reveal);addEventListener("DOMContentLoaded",reveal);</script>','</html>']
    (output/'questions'/f'{q}.html').write_text('\n'.join(content))
