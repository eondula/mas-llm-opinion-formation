"""Complete, inference-free reports and figures for recorded StrategyQA experiments."""
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from itertools import combinations
from pathlib import Path
import zipfile
from .behavioral import analyze, average, fraction, csv_write
from .storage import write

VERSION = 'strategyqa-reports-v1'
CONDITION = ('models','agents','schedule','initialization','decoding','research_split','source_partition')


def key(row):
    return tuple(row.get(k) for k in CONDITION)


def fmt(value):
    if value is None:return '—'
    if isinstance(value,float):return f'{value:.4f}'
    return str(value)


def table(rows, columns):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(label)+'</th>' for _,label in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(fmt(row.get(field)))+'</td>' for field,_ in columns)+'</tr>' for row in rows)+'</tbody></table></div>'


def document(title, body):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+'</title><style>body{font:15px/1.6 system-ui,sans-serif;color:#1b2337;background:#f4f5f9;margin:0}main{max-width:1400px;margin:auto;padding:30px}section{background:white;border:1px solid #d9deea;border-radius:12px;padding:22px;margin:22px 0}h1,h2,h3{line-height:1.3}a{color:#2563eb}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:9px;text-align:left;border-bottom:1px solid #d9deea;vertical-align:top}th{background:#eceef5}.scroll{overflow-x:auto}p{max-width:1100px}code{overflow-wrap:anywhere}.note{background:#eceef5;padding:14px;border-radius:8px}</style><main><h1>'+escape(title)+'</h1>'+body+'</main></html>'


def snapshots(path):
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((path/'conversations').glob('*.json'))}


def round_summary(data, conditions):
    rows=[]
    for kind in ('turns','groups'):
        buckets=defaultdict(list)
        for row in data[kind]:buckets[(key(row),row['round'])].append(row)
        for (condition,t),points in sorted(buckets.items(),key=lambda p:str(p[0])):
            fields=dict(condition=conditions[condition],round=t,unit='agent_response' if kind=='turns' else 'conversation',observations=len(points))
            if kind=='turns':
                labeled=[r for r in points if r['labeled']]
                fields.update(accuracy=fraction([r['correct'] is True for r in labeled]),
                    reversal_rate=fraction([r['answer_reversal'] for r in points]),stable_rate=fraction([r['stable'] for r in points]),
                    answer_oscillation_rate=fraction([r['answer_oscillation'] for r in points]),direction_reversal_rate=fraction([r['direction_reversal'] for r in points]),
                    mean_stance=average([r['stance'] for r in points]),mean_margin=average([r['margin'] for r in points]))
            else:
                labeled=[r for r in points if r['labeled']]
                fields.update(disagreement_rate=fraction([r['disagreement'] for r in points]),
                    consensus_rate=fraction([r['consensus'] for r in points]),
                    majority_accuracy=fraction([r['majority_correct'] is True for r in labeled]),
                    consensus_accuracy=fraction([r['consensus_correct'] for r in points if r['consensus']]),
                    mean_dispersion=average([r['stance_dispersion'] for r in points]))
            rows.append(fields)
    return rows


def generate(path, reanalyze=True):
    """Only read recorded observations; do not load generation or embedding models."""
    path=Path(path)
    before=snapshots(path)
    if reanalyze:analyze(path)
    compact=(path/'config.json').exists() and json.loads((path/'config.json').read_text()).get('question_figure_profile')=='question_png'
    if not compact:
        from .behavioral import plots
        plots(path)
    data=json.loads((path/'analysis/behavioral.json').read_text())
    references={}
    questions={}
    for filename in before:
        record=json.loads((path/filename).read_text())
        references[record['question_id']]=record.get('evaluation',{}).get('reference_answer')
        questions[record['question_id']]=record.get('task',record.get('scenario',{})).get('question','')
    distribution={answer:sum(value==answer for value in references.values()) for answer in ('Yes','No')}
    distribution['unlabeled']=sum(value not in ('Yes','No') for value in references.values())
    turns,agents,groups=data['turns'],data['agents'],data['groups']
    from .pilot_analysis import analyze as pilot_analysis
    pilot_analysis(path,data,make_figures=not compact)
    conditions={k:f'C{i+1}' for i,k in enumerate(sorted({key(r) for r in turns},key=str))}
    condition_rows=[dict(condition=label,**dict(zip(CONDITION,k))) for k,label in conditions.items()]
    csv_write(path/'tables/report_conditions.csv',condition_rows)
    # Per-agent aggregates avoid hiding differences between heterogeneous models.
    buckets=defaultdict(list)
    for row in agents:buckets[(key(row),row['agent'],row['model'])].append(row)
    agent_rows=[]
    for (condition,name,model),points in buckets.items():
        labeled=[r for r in points if r['labeled']]
        agent_rows.append(dict(condition=conditions[condition],agent=name,model=model,trajectories=len(points),
            initial_accuracy=fraction([r['initial_correct'] is True for r in labeled]),
            final_accuracy=fraction([r['final_correct'] is True for r in labeled]),
            reversals=sum(r['answer_reversals'] for r in points),correct_to_incorrect=sum(r['correct_to_incorrect'] for r in points),
            incorrect_to_correct=sum(r['incorrect_to_correct'] for r in points),answer_oscillations=sum(r['answer_oscillations'] for r in points),
            direction_reversals=sum(r['direction_reversals'] for r in points)))
    csv_write(path/'tables/per_agent_summary.csv',agent_rows)
    rounds=round_summary(data,conditions)
    # CSV rows have different available fields for the two explicitly identified units.
    csv_write(path/'tables/round_summary.csv',[{k:r.get(k) for k in dict.fromkeys(k for r in rounds for k in r)} for r in rounds])
    question_buckets=defaultdict(list)
    for row in agents:question_buckets[(key(row),row['question_id'])].append(row)
    question_rows=[]
    for (condition,qid),points in question_buckets.items():
        labeled=[r for r in points if r['labeled']]
        question_rows.append(dict(condition=conditions[condition],question_id=qid,question=questions.get(qid),reference_answer=references.get(qid),runs=len({r['repetition'] for r in points}),
            initial_accuracy=fraction([r['initial_correct'] is True for r in labeled]),final_accuracy=fraction([r['final_correct'] is True for r in labeled]),
            reversals=sum(r['answer_reversals'] for r in points)))
    csv_write(path/'tables/per_question_summary.csv',question_rows)
    if compact:
        from .question_figures import report
        report(path,data)
        if snapshots(path)!=before:raise ValueError('Report generation changed saved conversations')
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .plot_data import export as export_plot_data
    output=path/'figures/behavioral';output.mkdir(parents=True,exist_ok=True)
    figures=[]
    def finish(fig,stem,title,note):
        for ax in fig.axes:
            ax.grid(alpha=.18)
            handles,labels=ax.get_legend_handles_labels()
            if handles:ax.legend(fontsize=7)
            finite_lines=any(any(math.isfinite(float(y)) for y in line.get_ydata()) for line in ax.lines)
            if not finite_lines and not ax.collections and not ax.texts:
                ax.text(.5,.5,'No applicable observations',ha='center',va='center',transform=ax.transAxes)
        fig.suptitle(title,fontsize=13)
        for ext in ('png','pdf'):fig.savefig(output/f'{stem}.{ext}',dpi=150,bbox_inches='tight')
        export_plot_data(fig,output/f'{stem}.csv')
        if not (output/f'{stem}.csv').exists():csv_write(output/f'{stem}.csv',[])
        plt.close(fig)
        figures.append(dict(stem=stem,title=title,note=note,png=f'figures/behavioral/{stem}.png',pdf=f'figures/behavioral/{stem}.pdf',data=f'figures/behavioral/{stem}.csv'))
    def line(ax, source, value, unit='agent', reducer=average, only_updates=False):
        sets=defaultdict(list)
        for row in source:
            if only_updates and row['round']==0:continue
            label=conditions[key(row)]+(' · '+row['agent']+' · '+row['model'].split('/')[-1] if unit=='agent' else '')
            sets[(label,row['round'])].append(row)
        curves=defaultdict(list)
        for (label,t),points in sets.items():curves[label].append((t,reducer([value(r) for r in points])))
        for label,points in sorted(curves.items()):
            points.sort();ax.plot([p[0] for p in points],[p[1] if p[1] is not None else math.nan for p in points],marker='o',markersize=3,label=label)
        ax.set_xlabel('Round (0 = initialization)');ax.set_xticks(sorted({r['round'] for r in source if not only_updates or r['round']>0}))
    with plt.rc_context({'font.size':9,'axes.labelweight':'normal','axes.titleweight':'normal','lines.linewidth':1.7}):
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        for row in agent_rows:
            ax.plot([0,1],[row['initial_accuracy'] if row['initial_accuracy'] is not None else math.nan,row['final_accuracy'] if row['final_accuracy'] is not None else math.nan],marker='o',label=row['condition']+' · '+row['agent']+' · '+row['model'].split('/')[-1])
        ax.set(xticks=[0,1],xticklabels=['Initial','Final'],ylabel='Accuracy',ylim=(-.05,1.05))
        finish(fig,'agent_correctness','Initial and final correctness by agent','All labeled trajectories; invalid answers count as unsuccessful.')
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for ax,field,title in zip(axes.flat,['stance','margin','stance_change','margin_change'],['Post-response stance','Post-response likelihood margin','Stance change from previous turn','Margin change from previous turn']):
            line(ax,turns,lambda r:r[field],only_updates=field.endswith('change'));ax.set(title=title,ylabel='Mean '+field.replace('_',' '));ax.axhline(0,color='gray',lw=.7)
        finish(fig,'stance_margin_changes','Stance and likelihood-margin evolution','Means over questions/repetitions for each agent. Cross-model score scales are not calibrated.')
        fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
        line(axes[0],turns,lambda r:r['answer_reversal'],reducer=fraction,only_updates=True);axes[0].set(title='Answer reversal rate',ylabel='Fraction of adjacent valid answers',ylim=(-.05,1.05))
        line(axes[1],turns,lambda r:r['answer_oscillation'],reducer=fraction,only_updates=True);axes[1].set(title='Answer oscillation rate (A→B→A)',ylabel='Fraction of eligible three-turn windows',ylim=(-.05,1.05))
        finish(fig,'answer_reversals','Categorical answer changes','Invalid outputs break adjacent comparisons; initialization has no reversal.')
        fig,ax=plt.subplots(figsize=(11,5),layout='constrained');categories=['correct_to_correct','correct_to_incorrect','incorrect_to_correct','incorrect_to_incorrect']
        for condition,label in conditions.items():
            points=[r for r in turns if key(r)==condition]
            ax.plot(range(4),[sum(r['correctness_transition']==category for r in points) for category in categories],marker='o',label=label)
        ax.set(xticks=range(4),xticklabels=['Correct → correct','Correct → incorrect','Incorrect → correct','Incorrect → incorrect'],ylabel='Valid labeled transitions')
        finish(fig,'correctness_transitions','Correctness transition counts','Counts of adjacent valid labeled responses, not independent statistical samples.')
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        for ax,field,title in zip(axes,['stable','direction_reversal','answer_oscillation'],['Stable stance updates','Stance-direction reversals','Answer oscillations']):
            line(ax,turns,lambda r:r[field],reducer=fraction,only_updates=True);ax.set(title=title,ylabel='Fraction of eligible updates',ylim=(-.05,1.05))
        finish(fig,'stability_oscillation','Stability and oscillation','Stance tolerance 0.01; flat updates break consecutive direction reversals. Missing windows are excluded.')
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for ax,field,title in zip(axes.flat,['disagreement','pairwise_answer_disagreement','stance_dispersion','consensus'],['Group answer disagreement','Pairwise answer disagreement','Stance dispersion (population SD)','Consensus fraction']):
            line(ax,groups,lambda r:r[field],unit='group');ax.set(title=title,ylabel=title)
            if field!='stance_dispersion':ax.set_ylim(-.05,1.05)
        finish(fig,'group_dynamics','Group disagreement, dispersion and consensus','Disagreement requires complete valid multi-agent groups. Consensus is undefined for one agent.')
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        functions=[lambda r:(r['consensus_correct'] is True) if r['labeled'] and r['agents']>1 else None,
                   lambda r:(r['consensus_correct'] is False) if r['labeled'] and r['agents']>1 else None,
                   lambda r:r['consensus_correct'] if r['consensus'] else None]
        for ax,function,title in zip(axes,functions,['Correct consensus / labeled groups','Incorrect consensus / labeled groups','Accuracy among observed consensuses']):
            line(ax,groups,function,unit='group',reducer=fraction);ax.set(title=title,ylabel='Fraction',ylim=(-.05,1.05))
        finish(fig,'consensus_correctness','Consensus correctness','The first two panels use all labeled multi-agent groups; the third conditions on consensus.')
        fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
        for condition,label in conditions.items():
            points=[r for r in groups if key(r)==condition and r['labeled']]
            ys=[fraction([r['majority_correct'] is True for r in points if r[phase]]) for phase in ('initial','final')]
            axes[0].plot([0,1],[y if y is not None else math.nan for y in ys],marker='o',label=label)
        axes[0].set(xticks=[0,1],xticklabels=['Initial','Final'],ylabel='Majority accuracy',ylim=(-.05,1.05))
        line(axes[1],groups,lambda r:(r['majority_correct'] is True) if r['labeled'] else None,unit='group',reducer=fraction);axes[1].set(title='Majority accuracy across rounds',ylabel='Accuracy',ylim=(-.05,1.05))
        finish(fig,'majority_correctness','Initial, final and round-wise majority correctness','All labeled conversations; ties and invalid/incomplete groups count as unsuccessful.')
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        ticks=sorted({r['round'] for r in groups});positions=list(range(len(ticks)+1))
        for condition,label in conditions.items():
            points=[r for r in groups if key(r)==condition and r['final'] and r['agents']>1]
            ys=[sum(r['first_consensus_round']==t for r in points) for t in ticks]+[sum(r['first_consensus_round'] is None for r in points)]
            ax.plot(positions,[n/len(points) if points else math.nan for n in ys],marker='o',label=label)
        ax.set(xticks=positions,xticklabels=[str(t) for t in ticks]+['Never'],xlabel='First consensus round',ylabel='Fraction of multi-agent conversations',ylim=(-.05,1.05))
        finish(fig,'consensus_timing','When consensus first occurs','Round 0 means consensus already existed before peer interaction; first occurrence does not imply persistence.')
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        line(axes[0],turns,lambda r:r['peer_distance_reduction'],only_updates=True);axes[0].set(title='Distance reduction to delivered peers',ylabel='Positive = closer to delivered peer mean');axes[0].axhline(0,color='gray',lw=.7)
        for condition,label in conditions.items():
            points=[r for r in turns if key(r)==condition and r['peer_mean_stance'] is not None and r['previous_stance'] is not None and r['movement_toward_peers'] is not None]
            if points:axes[1].scatter([r['peer_mean_stance']-r['previous_stance'] for r in points],[r['stance_change'] for r in points],s=18,alpha=.7,label=label)
        axes[1].set(title='Delivered-peer gap versus own update',xlabel='Peer mean − previous own stance',ylabel='Own stance change');axes[1].axhline(0,color='gray',lw=.7);axes[1].axvline(0,color='gray',lw=.7)
        line(axes[2],turns,lambda r:r['delivery_stance_change'],only_updates=True);axes[2].set(title='Change immediately after delivery',ylabel='Decision-context stance − before delivery');axes[2].axhline(0,color='gray',lw=.7)
        finish(fig,'peer_movement','Associations with messages actually delivered','Uses recorded source IDs and synchronous/sequential timing. These are descriptive, not causal effects; cross-model scores may have different scales.')
        import csv
        with (path/'tables/no_peer_comparisons.csv').open() as stream:controls=list(csv.DictReader(stream))
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        if controls:
            sets=defaultdict(list)
            for row in controls:sets[tuple(str(row.get(k)) for k in CONDITION)].append(row)
            labels=[]
            for i,(control_key,points) in enumerate(sets.items()):
                label=next((name for k,name in conditions.items() if tuple(str(v) for v in k)==control_key),str(control_key))
                labels.append(label)
                for ax,field in zip(axes,['final_stance_difference','final_correctness_difference']):
                    values=[float(r[field]) for r in points if r.get(field) not in ('',None)]
                    if values:ax.plot([i],[average(values)],marker='o',label=label)
            for ax in axes:ax.set_xticks(range(len(sets)),labels)
        else:
            for ax in axes:ax.text(.5,.5,'Unavailable: no matched no-peer control recorded',ha='center',va='center',wrap=True,transform=ax.transAxes)
        axes[0].set(title='Matched final stance difference',ylabel='Deliberation − no peer');axes[1].set(title='Matched final accuracy difference',ylabel='Deliberation − no peer')
        finish(fig,'no_peer_comparison','Matched no-peer comparisons','Matched by question, repetition, agent and experimental settings. Missing controls are not imputed.')
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        for ax,field,title in zip(axes,['adjacent_message_cosine','initial_message_cosine','exact_delivered_peer_argument'],['Cosine to previous own message','Cosine to initial own message','Exact reuse of a delivered peer argument']):
            line(ax,turns,lambda r:r[field],only_updates=True);ax.set(title=title,ylabel='Mean cosine' if 'cosine' in field else 'Fraction',ylim=(-1.05,1.05) if 'cosine' in field else (-.05,1.05))
        finish(fig,'message_reuse','Message similarity and peer argument reuse','Cosine uses full public messages; exact reuse compares explanation text. Initial/invalid/unresolved comparisons are unavailable.')
    manifest=dict(version=VERSION,generated_utc=datetime.now(timezone.utc).isoformat(),source_sha256=before,figures=figures,
                  controls=dict(available=bool(controls),matched_rows=len(controls)),conditions=condition_rows,label_distribution=distribution,
                  definitions=data['definitions'])
    write(path/'analysis/report_manifest.json',manifest)
    body='<p class="note">Descriptive results from saved trajectories. No model inference was rerun. Peer movement is not a causal estimate. Scores across different models are not calibrated.</p>'
    body+='<p><a href="REPORT.md">Text report</a> · <a href="summary.json">Summary JSON</a> · <a href="paper_outputs.zip">Download reports, figures and tables ZIP</a> · <a href="figures/index.html">All figures</a></p>'
    if path.parent.name=='experiments':body+='<p><a href="../comparisons/index.html">Cross-experiment comparison</a></p>'
    body+='<section><h2>Question label distribution</h2><p>'+escape(str(distribution))+'</p><p>Counts are unique questions, not repetitions. A single-label selection cannot establish balanced Yes/No performance.</p></section>'
    body+='<section><h2>Experimental conditions</h2>'+table(condition_rows,[('condition','Condition'),('models','Models'),('agents','Agents'),('schedule','Schedule'),('initialization','Start'),('decoding','Decoding'),('research_split','Split')])+'</section>'
    body+='<section><h2>Condition summaries</h2>'+table(data['summary'],[('models','Models'),('schedule','Schedule'),('initial_accuracy','Initial accuracy'),('final_accuracy','Final accuracy'),('initial_majority_accuracy','Initial majority accuracy'),('final_majority_accuracy','Final majority accuracy'),('final_consensus_rate','Final consensus'),('final_consensus_accuracy','Consensus accuracy'),('answer_reversals','Reversals'),('correct_to_incorrect','Correct→incorrect'),('incorrect_to_correct','Incorrect→correct'),('stable_update_fraction','Stable fraction'),('answer_oscillations','Oscillations'),('exact_peer_argument_reuse_rate','Peer argument reuse')])+'</section>'
    body+='<section><h2>Per-agent outcomes</h2>'+table(agent_rows,[(k,k.replace('_',' ').title()) for k in agent_rows[0]])+'</section>'
    body+='<section><h2>Per-question outcomes</h2>'+table(question_rows,[(k,k.replace('_',' ').title()) for k in question_rows[0]])+'</section>'
    body+='<section><h2>Detailed exports</h2><ul>'+''.join('<li><a href="'+escape(str(f.relative_to(path)))+'">'+escape(f.name)+'</a></li>' for folder in ['tables','analysis','audit'] for f in sorted((path/folder).glob('*')) if f.suffix in ('.csv','.json'))+'</ul></section>'
    body+='<section><h2>Pilot validation</h2><p><a href="analysis/pilot_validation.json">Pairing and variation diagnostics</a> · <a href="figures/informative_variation.pdf">Variation PDF</a></p><img src="figures/informative_variation.png" alt="Initial stance, peer gaps and update variation"></section>'
    if (path/'audit/stance_audit.json').exists():body+='<section><h2>Stance audit</h2><p><a href="audit/stance_audit.json">Raw probes and diagnostics</a> · <a href="audit/summary.csv">Audit summary</a> · <a href="audit/stance_audit.pdf">PDF</a></p><img src="audit/stance_audit.png" alt="Identical history stance audit"></section>'
    body+='<section><h2>Control availability</h2><p>'+('Matched no-peer rows: '+str(len(controls)) if controls else 'No matched no-peer control was recorded. Additional control runs are required; existing peer trajectories cannot supply that counterfactual.')+'</p></section>'
    for entry in figures:
        body+='<section><h2>'+escape(entry['title'])+'</h2><p>'+escape(entry['note'])+'</p><p><a href="'+entry['pdf']+'">PDF</a> · <a href="'+entry['data']+'">Plotted data CSV</a></p><img loading="lazy" alt="'+escape(entry['title'])+'" src="'+entry['png']+'"></section>'
    (path/'index.html').write_text(document('Experiment report · '+path.name,body))
    report=['# Experiment report: '+path.name,'','These are descriptive results from recorded trajectories. No generation or embedding inference was rerun.','',
            '## Conditions','',*[f'- {r["condition"]}: {r["models"]}; {r["agents"]} agents; {r["schedule"]}; start {r["initialization"]}; {r["decoding"]}; {r["research_split"]}.' for r in condition_rows],
            '', '## Summary','', '| Condition | Initial accuracy | Final accuracy | Final consensus | Answer reversals | Correct→incorrect | Incorrect→correct |','|---|---:|---:|---:|---:|---:|---:|']
    for r in data['summary']:report.append('| '+' | '.join(fmt(v) for v in [conditions[key(r)],r['initial_accuracy'],r['final_accuracy'],r['final_consensus_rate'],r['answer_reversals'],r['correct_to_incorrect'],r['incorrect_to_correct']])+' |')
    report+=['','## Reports and figures','',*[f'- [{r["title"]}]({r["png"]}) · [PDF]({r["pdf"]}) · [data]({r["data"]}). {r["note"]}' for r in figures],
             '', '## Interpretation limits','', 'Accuracy includes invalid labeled answers as unsuccessful; majority ties also count as unsuccessful. Consensus is undefined for one agent. Natural and forced starts remain separate. Repetitions, agents and rounds from the same question are not independent questions. No uncertainty intervals or significance tests are claimed. Cross-model stance scores are not calibrated. Peer movement and matched-control differences are descriptive. Missing labels and missing controls remain unavailable.',
             '', 'Unique-question reference labels: '+str(distribution)+'. Single-label selections do not establish balanced Yes/No performance.',
             '', 'Matched no-peer observations: '+str(len(controls))+'.', '', 'See [the complete HTML report](index.html), [round summaries](tables/round_summary.csv), [per-agent summaries](tables/per_agent_summary.csv), [per-question summaries](tables/per_question_summary.csv), and [delivery provenance](tables/peer_deliveries.csv).']
    (path/'REPORT.md').write_text('\n'.join(report)+'\n')
    # Update the old figure entry point after every new figure exists.
    gallery='<p><a href="../index.html">Complete experiment report</a> · <a href="../tables/summary.csv">Summary CSV</a></p>'
    for f in sorted((path/'figures').rglob('*.png')):
        rel=str(f.relative_to(path/'figures'));pdf=f.with_suffix('.pdf')
        gallery+='<section><h2>'+escape(rel)+'</h2>'+(f'<a href="{escape(str(pdf.relative_to(path/"figures")))}">PDF</a>' if pdf.exists() else '')+'<img loading="lazy" alt="'+escape(rel)+'" src="'+escape(rel)+'"></section>'
    (path/'figures/index.html').write_text(document('All experiment figures · '+path.name,gallery))
    coverage=path/'figures/coverage.json'
    old=json.loads(coverage.read_text()) if coverage.exists() else {}
    old['behavioral_reports']=dict(version=VERSION,figures=len(figures),no_peer_control_available=bool(controls))
    if isinstance(old.get('pending'),list):old['pending']=[v for v in old['pending'] if v!='group_summary']
    write(coverage,old)
    temporary=path/'paper_outputs.zip.tmp'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ['figures','tables','analysis','audit','initializations']:
            for f in sorted((path/folder).rglob('*')):
                if f.is_file():archive.write(f,f.relative_to(path))
        for name in ['index.html','REPORT.md','summary.json','RESULTS.md']:
            if (path/name).exists():archive.write(path/name,name)
    temporary.replace(path/'paper_outputs.zip')
    if snapshots(path)!=before:raise RuntimeError('Report generation changed a recorded conversation')
    return manifest


def compare_experiments(parent, include_current=None):
    """Descriptive cross-experiment exports; pair questions only for matched protocols."""
    parent=Path(parent);experiments=[]
    for path in sorted(parent.iterdir()) if parent.exists() else []:
        if not (path/'analysis/behavioral.json').exists() or not (path/'config.json').exists():continue
        status=json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {}
        if status.get('phase')!='complete' and path!=include_current:continue
        data=json.loads((path/'analysis/behavioral.json').read_text());config=json.loads((path/'config.json').read_text())
        prompts=(path/'inputs/prompts.json').read_bytes() if (path/'inputs/prompts.json').exists() else b''
        protocol={k:config.get(k) for k in ['pilot_protocol','paired_control','model_revisions','expected_dtype','device','question_ids','dataset','rounds','agents','repetitions','schedule','conditions','initialization','decoding','sampling','seed','max_argument_tokens','stopping']}
        protocol['prompt_sha256']=hashlib.sha256(prompts).hexdigest()
        signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        experiments.append(dict(path=path,data=data,signature=signature))
    if not experiments:return None
    folder=parent/'comparisons';folder.mkdir(exist_ok=True)
    summaries=[];pair_rows=[];curve_rows=[]
    for i,item in enumerate(experiments):
        item['label']=f'E{i+1}'
        for summary in item['data']['summary']:summaries.append(dict(experiment=item['path'].name,label=item['label'],protocol_signature=item['signature'],**summary))
        buckets=defaultdict(list)
        for row in item['data']['groups']:buckets[(key(row),row['round'])].append(row)
        for (condition,t),points in buckets.items():
            curve_rows.append(dict(experiment=item['path'].name,label=item['label'],round=t,**dict(zip(CONDITION,condition)),
                                   accuracy=average([r['agent_accuracy'] for r in points]),
                                   consensus=fraction([r['consensus'] for r in points]),dispersion=average([r['stance_dispersion'] for r in points]),
                                   cosine=average([r['mean_adjacent_message_cosine'] for r in points])))
    for left,right in combinations(experiments,2):
        if left['signature']!=right['signature']:continue
        def question_scores(item):
            buckets=defaultdict(list)
            for row in item['data']['agents']:
                # Different schedules and starts remain separate within matched multi-condition batches.
                buckets[(row['question_id'],row['schedule'],row['initialization'],row['agents'],row['research_split'])].append(row)
            return {k:dict(initial=fraction([r['initial_correct'] is True for r in rows if r['labeled']]),final=fraction([r['final_correct'] is True for r in rows if r['labeled']])) for k,rows in buckets.items()}
        a,b=question_scores(left),question_scores(right)
        for k in sorted(a.keys()&b.keys(),key=str):
            pair_rows.append(dict(experiment_a=left['path'].name,experiment_b=right['path'].name,question_id=k[0],schedule=k[1],initialization=k[2],agents=k[3],research_split=k[4],
                                  initial_accuracy_a=a[k]['initial'],initial_accuracy_b=b[k]['initial'],
                                  final_accuracy_a=a[k]['final'],final_accuracy_b=b[k]['final'],
                                  final_accuracy_difference=b[k]['final']-a[k]['final'] if a[k]['final'] is not None and b[k]['final'] is not None else None))
    csv_write(folder/'summary.csv',summaries);csv_write(folder/'paired_question_comparisons.csv',pair_rows);csv_write(folder/'round_comparisons.csv',curve_rows)
    write(folder/'comparison.json',dict(summary=summaries,paired_questions=pair_rows,rounds=curve_rows,
                                      note='Descriptive cross-experiment comparisons. Pairing requires identical recorded protocol, dataset selection and prompt templates apart from model assignment. No significance tests or causal claims.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .plot_data import export as export_plot_data
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    curves=defaultdict(list)
    for r in curve_rows:curves[(r['label'],key(r))].append(r)
    for (label,condition),rows in curves.items():
        rows.sort(key=lambda r:r['round'])
        legend=label+' · '+str(condition[1])+' agents · '+condition[2]+' · '+condition[3]
        for ax,field in zip(axes.flat,['accuracy','consensus','dispersion','cosine']):ax.plot([r['round'] for r in rows],[r[field] if r[field] is not None else math.nan for r in rows],marker='o',label=legend)
    for ax,title in zip(axes.flat,['Agent accuracy','Consensus fraction','Stance dispersion','Adjacent-message cosine']):
        ax.set(xlabel='Round',ylabel=title,title=title);ax.set_xticks(sorted({r['round'] for r in curve_rows}));ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Descriptive experiment comparison')
    for ext in ('png','pdf'):fig.savefig(folder/f'comparison.{ext}',dpi=150,bbox_inches='tight')
    export_plot_data(fig,folder/'comparison_plot_data.csv');plt.close(fig)
    body='<p class="note">Descriptive comparison. Matched-question exports require the same recorded protocol and prompts. Different model score scales can affect stance dispersion. No confidence intervals, significance tests or causal effects are claimed.</p>'
    body+='<section><h2>Experiment summaries</h2>'+table(summaries,[('label','Label'),('experiment','Experiment'),('models','Models'),('initial_accuracy','Initial accuracy'),('final_accuracy','Final accuracy'),('final_consensus_rate','Final consensus'),('mean_final_dispersion','Final dispersion'),('exact_peer_argument_reuse_rate','Peer reuse')])+'</section>'
    body+='<section><h2>Reports</h2><ul>'+''.join('<li><a href="../'+escape(item['path'].name)+'/index.html">'+escape(item['label']+' · '+item['path'].name)+'</a></li>' for item in experiments)+'</ul></section>'
    body+='<section><h2>Figures and exports</h2><p><a href="comparison.pdf">PDF</a> · <a href="summary.csv">Summary CSV</a> · <a href="round_comparisons.csv">Round comparisons</a> · <a href="paired_question_comparisons.csv">Matched-question CSV</a> · <a href="comparison.json">JSON</a></p><img src="comparison.png" alt="Comparison of accuracy, consensus, dispersion and message cosine"></section>'
    (folder/'index.html').write_text(document('Experiment comparisons',body))
    return folder


def rebuild_all(root):
    """Rebuild completed active StrategyQA results; historical archives are untouched."""
    from .locations import run_directories, historical_root
    rebuilt=[]
    for path in run_directories(root):
        if path.is_relative_to(historical_root(root)):continue
        config=json.loads((path/'config.json').read_text()) if (path/'config.json').exists() else {}
        status=json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {}
        if not config.get('dataset') or status.get('phase')!='complete':continue
        if not list((path/'conversations').glob('*.json')):continue
        generate(path);rebuilt.append(str(path))
        print('Reports generated: '+str(path),flush=True)
    compare_experiments(Path(root)/'results/experiments')
    return rebuilt
