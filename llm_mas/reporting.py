"""Paper export manifest and aggregate figures, generated only from saved data."""
import json
import shutil
from pathlib import Path
from html import escape


def export(path):
    import matplotlib.pyplot as plt
    path=Path(path);folder=path/'figures'
    for name in ('main','appendix','data'):(folder/name).mkdir(exist_ok=True)
    records=[(p.stem,json.loads(p.read_text())) for p in sorted((path/'conversations').glob('*.json'))]
    produced=[]
    def save(fig,name,section):
        from .plot_data import export as export_plot_data
        if not name.endswith('_public_states'):export_plot_data(fig,folder/section/(name+'.csv'))
        for ext in ('pdf','png'):fig.savefig(folder/section/(name+'.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig);produced.append(dict(name=name,section=section))
    for page in range(0,len(records),6):
        subset=records[page:page+6]
        fig,axes=plt.subplots(len(subset),1,figsize=(11,3*len(subset)),squeeze=False,layout='constrained')
        for ax,(task,record) in zip(axes[:,0],subset):
            for agent in dict.fromkeys(r['agent'] for r in record['rows']):
                rows=[r for r in record['rows'] if r['agent']==agent]
                ax.plot([r['round'] for r in rows],[r['post']['stance'] for r in rows],label=agent)
            c=record['config'];ax.set(title=f"{task}: {record['scenario']['title']} / {c['initialization']} / {c['schedule']}",xlabel='Round',ylabel='Post-response stance',ylim=(-1.05,1.05));ax.axhline(0,color='gray',lw=.5);ax.legend()
        save(fig,f'stance_trajectories_{page//6+1:02d}','main')
        fig,axes=plt.subplots(len(subset),1,figsize=(11,3*len(subset)),squeeze=False,layout='constrained')
        for ax,(task,record) in zip(axes[:,0],subset):
            seen={};repetitions=[]
            for r in record['rows']:
                key=(r['agent'],r['message']);seen[key]=seen.get(key,0)+1;repetitions.append(seen[key])
            ax.plot(repetitions);ax.set(title=task,xlabel='Response index',ylabel='Occurrences of exact message')
        save(fig,f'message_repetition_{page//6+1:02d}','appendix')
    for task,record in records:
        import networkx as nx
        graph=nx.DiGraph();states={}
        for row in record['rows']:states.setdefault(row['round'],{})[row['agent']]=row['strategy']
        labels=[''.join(str(s[n]) for n in sorted(s)) for _,s in sorted(states.items())]
        graph.add_nodes_from(labels)
        for a,b in zip(labels,labels[1:]):graph.add_edge(a,b,weight=graph.get_edge_data(a,b,{}).get('weight',0)+1)
        fig,ax=plt.subplots(figsize=(6,5));positions=nx.circular_layout(graph)
        nx.draw_networkx(graph,pos=positions,ax=ax,node_color='#dbe8f5',node_size=1300,arrows=True,connectionstyle='arc3,rad=0.15')
        nx.draw_networkx_edge_labels(graph,pos=positions,edge_labels=nx.get_edge_attributes(graph,'weight'),ax=ax)
        from .tables import table
        table(folder/'appendix'/(task+'_public_states'),[dict(source=a,target=b,transitions=d['weight']) for a,b,d in graph.edges(data=True)])
        ax.set_title(task+' public states (edge labels = transition counts)');ax.axis('off');save(fig,task+'_public_states','appendix')
        row=next((r for r in record['rows'] if r['round']>0 and r['decision']),None)
        if row and row['decision'].get('token_log_probs'):
            fig,ax=plt.subplots(figsize=(9,4))
            for index,values in enumerate(row['decision']['token_log_probs']):ax.plot(range(len(values)),values,marker='.',label=f'Candidate {index+1}')
            ax.set(xlabel='Candidate token position (separate tokenizations)',ylabel='Conditional token log likelihood',title=task+' first decision token scores');ax.legend();save(fig,task+'_candidate_tokens','appendix')
    source=path/'analysis/prediction.json'
    if source.exists():
        data=json.loads(source.read_text())
        if data['rows']:
            fig,ax=plt.subplots(figsize=(6,5))
            ax.scatter([r['target'] for r in data['rows']],[r['prediction'] for r in data['rows']],label='Held-out conversation')
            ax.set(xlabel='Observed decision margin',ylabel='Predicted decision margin');ax.legend();save(fig,'prediction','main')
    for source in sorted(folder.glob('*')):
        if not source.is_file():continue
        if source.suffix in ('.csv','.json'):
            shutil.copy2(source,folder/'data'/source.name)
            if source.suffix=='.csv' and source.with_suffix('.png').exists():
                shutil.copy2(source,folder/'appendix'/source.name)
        elif source.suffix in ('.pdf','.png','.txt'):shutil.copy2(source,folder/'appendix'/source.name)
    for filename in ('replay','mechanistic'):
        source=path/'measurements'/f'{filename}.json'
        if not source.exists():continue
        data=json.loads(source.read_text())
        for i,row in enumerate(data):
            fig,ax=plt.subplots(figsize=(9,4))
            if filename=='replay':
                ax.plot(list(row['scores']),[s['margin'] for s in row['scores'].values()],marker='o')
            else:
                for component in ('residual','attention','mlp'):
                    effects=[r for r in row['effects'] if r['component']==component and r['control']=='donor' and r['recipient']==0]
                    ax.plot([r['layer'] for r in effects],[r['delta'] for r in effects],label=component)
                ax.legend();ax.set_xlabel('Layer')
            ax.set_title(f"{row['task']} / {row['agent']} / round {row['round']}")
            ax.set_ylabel('Raw margin' if filename=='replay' else 'Margin change under intervention')
            save(fig,f'{filename}_{i:03d}','main' if i<2 else 'appendix')
    for filename in ('heads','source_spans','sufficiency','continuation'):
        source=path/'measurements'/f'{filename}.json'
        if not source.exists():continue
        data=json.loads(source.read_text())
        for i,row in enumerate(data):
            if filename=='sufficiency' and row.get('status')!='matched':continue
            fig,ax=plt.subplots(figsize=(9,4))
            if filename in ('heads','source_spans'):
                effects=row['effects']
                if filename=='heads':
                    import numpy as np
                    selected=[r for r in effects if r['recipient']==1]
                    matrix=np.full((max(r['layer'] for r in selected)+1,max(r['head'] for r in selected)+1),np.nan)
                    for r in selected:matrix[r['layer'],r['head']]=r['delta']
                    im=ax.imshow(matrix,aspect='auto',cmap='coolwarm');fig.colorbar(im,ax=ax,label='Margin change');ax.set(xlabel='Head',ylabel='Layer')
                else:ax.scatter([r['layer'] for r in effects],[r['delta'] for r in effects],c=[r['span'] for r in effects],cmap='viridis')
                ax.set(xlabel='Layer',ylabel='Raw margin change')
            elif filename=='sufficiency':
                for j in range(2):ax.plot(['Before','After same message'],[row['before'][j],row['after'][j]],marker='o',label=f'History {j+1}')
                ax.set_ylabel('Raw margin');ax.legend()
            else:
                for name in dict.fromkeys(r['agent'] for r in row['rows']):
                    rs=[r for r in row['rows'] if r['agent']==name];bs=[r for r in row['baseline'] if r['agent']==name]
                    ax.plot([r['round'] for r in rs],[r['post']['margin'] for r in rs],label=name+' patched')
                    ax.plot([r['round'] for r in bs],[r['margin'] for r in bs],linestyle='--',label=name+' baseline')
                ax.axvline(row['round'],color='gray');ax.set(xlabel='Round',ylabel='Post-response raw margin');ax.legend()
            ax.set_title(f"{filename}: {row['task']} / {row['agent']} / round {row['round']}" + (f" / layer {row['layer']}" if filename=='continuation' else ''))
            save(fig,f'{filename}_{i:03d}','appendix')
    fig,ax=plt.subplots(figsize=(11,2));ax.axis('off')
    steps=['History','Candidate scores','Strategy','Public message','Peer delivery']
    for i,step in enumerate(steps):
        ax.text(.1+i*.2,.5,step,ha='center',va='center',bbox=dict(boxstyle='round',facecolor='#eaf0f8'))
        if i<4:ax.annotate('',xy=(.19+i*.2,.5),xytext=(.15+i*.2,.5),arrowprops={'arrowstyle':'->'})
    save(fig,'framework','main')
    from .tables import export as export_tables
    export_tables(path)
    config=json.loads((path/'config.json').read_text()) if (path/'config.json').exists() else {}
    requested=config.get('research',[])
    coverage={'produced':produced,'modules':{},'note':'Completion describes executed configured analyses, not scientific validation or positive results.'}
    for module in ('validation','history','replay','prediction','mechanistic','heads','source_spans','continuation','sufficiency'):
        source=path/('analysis' if module=='prediction' else 'measurements')/(module+'.json')
        coverage['modules'][module]='available' if source.exists() else 'missing' if module in requested else 'not_requested'
    coverage['complete_requested_suite']=all(v!='missing' for v in coverage['modules'].values())
    coverage['complete_paper_suite']=all(v=='available' for v in coverage['modules'].values())
    if (path/'measurements/validation.json').exists():
        checks=json.loads((path/'measurements/validation.json').read_text())
        coverage['independent_unparsed']=sum(e.get('parsed_choice') is None for r in checks for e in r.get('independent',[]))
        coverage['paraphrases']='available' if checks and all(r.get('variants') for r in checks) else 'not_supplied'
        if coverage['paraphrases']=='not_supplied':coverage['complete_paper_suite']=False
    (folder/'coverage.json').write_text(json.dumps(coverage,indent=2))
    links=[]
    for p in sorted(folder.rglob('*.png')):
        rel=p.relative_to(folder);links.append(f'<h2>{escape(str(rel))}</h2><a href="{escape(str(rel.with_suffix(".pdf")))}">PDF</a><img style="max-width:100%" src="{escape(str(rel))}">')
    (folder/'index.html').write_text('<!doctype html><title>Study figures</title><h1>Study figures</h1><p>See coverage.json for outstanding analyses.</p>'+''.join(links))

    import zipfile
    with zipfile.ZipFile(path/'paper_outputs.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for directory in ('figures','tables'):
            for item in (path/directory).rglob('*'):
                if item.is_file():archive.write(item,item.relative_to(path))
