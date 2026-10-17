"""Exploration exports. Plot regeneration reads only CSV files."""
import csv
import json
import platform
from pathlib import Path
from .exploration import references, names_for


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        out = csv.DictWriter(f, fieldnames=fields or ['response_id'])
        out.writeheader(); out.writerows(rows)


def analyze(path, encoder=None):
    from .message_similarity import MessageSimilarity
    path=Path(path); encoder=encoder or MessageSimilarity()
    tables={key:[] for key in ('responses','diagnostics','diagnostic_tokens','deliveries','argument_reference_similarity','argument_transition_similarity','peer_argument_similarity','round_outcomes','run_summary')}
    for file in sorted((path/'conversations').glob('*.json')):
        data=json.loads(file.read_text()); rows=data['rows']; config=data['config']; refs=references(data['scenario']); previous={}; by_id={r['response_id']:r for r in rows}
        for row in rows:
            common=dict(task=file.stem,response_id=row['response_id'],agent=row['agent'],round=row['round'])
            tables['responses'].append(dict(common,model=row['model'],exploration_setup=config.get('exploration_setup','standard'),scenario_family=data['scenario'].get('scenario_family',''),evidence_variant=data['scenario'].get('evidence_variant',''),information_access=config.get('information_access','none'),evidence_rotation=config.get('evidence_rotation',0),initial_evidence=json.dumps(row.get('initial_evidence',[])),persona=config.get('persona',False),priority=((names_for(config['agents']).index(row['agent'])+config.get('persona_rotation',0))%2+1) if config.get('persona') else '',argument_source=config['argument_source'],strategy=row['strategy'],forced_assignment=row['forced_assignment'],forced_compliance=row['forced_compliance'],validity=row['choice_status'],raw_output=row['generation']['output'],message=row['message'],explanation=row['argument'],argument_id=row['argument_id'],sentiment=row.get('sentiment'),decoding='greedy'))
            diag=row['diagnostic']; candidates=diag.get('candidates',[])
            tables['diagnostics'].append(dict(common,status=diag['status'],error=diag.get('error'),stage='post_response',score_s1=candidates[0]['score'] if candidates else None,score_s2=candidates[1]['score'] if candidates else None,margin=diag.get('margin'),stance=diag.get('stance'),implied_strategy=diag.get('strategy'),tie=diag.get('tie'),mismatch=(row['strategy']!=diag['strategy']) if row['strategy'] is not None and diag.get('strategy') is not None else None))
            for k, candidate in enumerate(candidates,1):
                for position,(token_id,token,score) in enumerate(zip(candidate['token_ids'],candidate['tokens'],candidate['token_scores'])):
                    tables['diagnostic_tokens'].append(dict(common,candidate=k,position=position,token_id=token_id,token=token,log_probability=score,**{key:candidate[key] for key in ('logits_dtype','log_softmax_dtype','gathered_dtype','accumulation_dtype')}))
            for delivery in row['deliveries']:
                tables['deliveries'].append(dict(common,**delivery))
            text=row['argument']
            if row['choice_status']=='valid':
                for ref,(strategy,argument) in refs.items():
                    tables['argument_reference_similarity'].append(dict(common,reference_id=ref,reference_strategy=strategy,reference_text=argument,similarity=encoder.compare([text],[argument])[0],exact=text==argument))
                prev=previous.get(row['agent'])
                if prev and prev['choice_status']=='valid':
                    tables['argument_transition_similarity'].append(dict(common,source_id=prev['response_id'],exact=text==prev['argument'],similarity=encoder.compare([text],[prev['argument']])[0],strategy_changed=row['strategy']!=prev['strategy']))
                for delivery in row['deliveries']:
                    peer=by_id[delivery['source_id']]
                    if peer['choice_status']=='valid':
                        tables['peer_argument_similarity'].append(dict(common,source_id=peer['response_id'],speaker=peer['agent'],similarity=encoder.compare([text],[peer['argument']])[0],exact=text==peer['argument']))
            previous[row['agent']]=row
        for t in sorted({r['round'] for r in rows}):
            group=[r for r in rows if r['round']==t]; valid=all(r['strategy'] in (1,2) for r in group)
            tables['round_outcomes'].append(dict(task=file.stem,round=t,choices=json.dumps({r['agent']:r['strategy'] for r in group}),valid_choices=sum(r['strategy'] in (1,2) for r in group),agents=len(group),agreement=len({r['strategy'] for r in group})==1 if valid else None))
        tables['run_summary'].append(dict(task=file.stem,responses=len(rows),valid_responses=sum(r['choice_status']=='valid' for r in rows),invalid_responses=sum(r['choice_status']!='valid' for r in rows),valid_transitions=sum(r['task']==file.stem for r in tables['argument_transition_similarity']),revisions=sum(r['task']==file.stem and r['strategy_changed'] for r in tables['argument_transition_similarity']),diagnostic_comparisons=sum(r['task']==file.stem and r['mismatch'] is not None for r in tables['diagnostics']),diagnostic_mismatches=sum(r['task']==file.stem and r['mismatch'] is True for r in tables['diagnostics'])))
    for name,rows in tables.items(): write_csv(path/(name+'.csv'),rows)
    (path/'analysis_metadata.json').write_text(json.dumps(dict(encoder=encoder.encoder,encoder_revision=encoder.revision,similarity_input='explanation only; chunked length-weighted embeddings',python=platform.python_version()),indent=2))


def plots(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=Path(path); folder=path/'figures';folder.mkdir(exist_ok=True)
    def read(name):
        with (path/(name+'.csv')).open() as f:return list(csv.DictReader(f))
    rows=read('responses'); diag={(r['task'],r['response_id']):r for r in read('diagnostics')}
    for task in sorted({r['task'] for r in rows}):
        group=[r for r in rows if r['task']==task]; agents=list(dict.fromkeys(r['agent'] for r in group))
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for i,agent in enumerate(agents):
            selected=[r for r in group if r['agent']==agent];color=f'C{i%10}'
            for ax,metric in zip(axes,('stance','strategy','sentiment')):
                valid=[(int(r['round']),diag[(task,r['response_id'])].get('stance','') if metric=='stance' else r[metric]) for r in selected]
                ax.plot([x for x,v in valid],[float(v) if v!='' else float('nan') for x,v in valid],marker='o',label=agent,color=color)
            for r in selected:
                if r['strategy']=='':axes[1].scatter(int(r['round']),1.5,marker='x',color=color)
                if r['forced_assignment']!='':axes[1].scatter(int(r['round']),float(r['forced_assignment']),marker='s',facecolors='none',edgecolors=color,s=100)
        for ax,title in zip(axes,('Diagnostic stance','Chosen strategy','Sentiment')):
            ax.set_title(title);ax.set_xlabel('Round');ax.grid(alpha=.2);ax.set_xlim(-.2,max(int(r['round']) for r in group)+.2)
        axes[0].set_ylim(-1.05,1.05);axes[2].set_ylim(-1.05,1.05);axes[1].set_yticks([1,2],['S1','S2']);axes[1].set_ylim(.7,2.3)
        if not any(diag[(task,r['response_id'])].get('stance') for r in group):axes[0].text(.5,.5,'Diagnostic disabled',transform=axes[0].transAxes,ha='center')
        axes[1].legend(fontsize=8);fig.savefig(folder/(task+'_trajectories.png'),dpi=160);plt.close(fig)
        transitions=[r for r in read('argument_transition_similarity') if r.get('task')==task]
        if transitions:
            fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
            for i,agent in enumerate(agents):
                points=[r for r in transitions if r['agent']==agent]
                ax.plot([int(r['round']) for r in points],[float(r['similarity']) for r in points],marker='o',label=agent,color=f'C{i%10}')
            ax.set(xlabel='Round',ylabel='Adjacent explanation cosine',ylim=(-1.05,1.05),title='Argument evolution');ax.legend();ax.grid(alpha=.2)
            fig.savefig(folder/(task+'_argument_evolution.png'),dpi=160);plt.close(fig)
        ref=read('argument_reference_similarity'); ref=[r for r in ref if r.get('task')==task]
        if ref:
            ids=list(dict.fromkeys(r['response_id'] for r in ref));keys=list(dict.fromkeys(r['reference_id'] for r in ref));lookup={(r['response_id'],r['reference_id']):float(r['similarity']) for r in ref}
            fig,ax=plt.subplots(figsize=(10,max(4,len(ids)*.18)),layout='constrained');im=ax.imshow([[lookup[(i,k)] for k in keys] for i in ids],vmin=-1,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks(range(len(keys)),keys,rotation=45);ax.set_yticks(range(len(ids)),ids);ax.set_title('Explanation similarity to reference arguments');fig.colorbar(im,ax=ax);fig.savefig(folder/(task+'_references.png'),dpi=160);plt.close(fig)
    (folder/'index.html').write_text('<!doctype html><title>Exploration figures</title>'+''.join(f'<p><img style="max-width:100%" src="{p.name}" alt="{p.stem}"></p>' for p in sorted(folder.glob('*.png'))))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('path');parser.add_argument('--plot-only',action='store_true');args=parser.parse_args()
    if not args.plot_only:analyze(args.path)
    plots(args.path)
