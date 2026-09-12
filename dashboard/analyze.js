'use strict';
const $=id=>document.getElementById(id);
let sourceData=null, job=null, polling=false, activeRun=null;
const option=(value,label)=>{const e=document.createElement('option');e.value=value;e.textContent=label;return e;};
async function api(path,data){const r=await fetch(path,data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{});const body=await r.json();if(!r.ok)throw Error(body.error||r.statusText);return body;}
function error(e){$('error').textContent=e.message;$('status').textContent='Analysis needs attention';}
function link(label,url){const a=document.createElement('a');a.textContent=label;a.href=url;a.setAttribute('download','');return a;}
function showProgress(status){
 const bar=$('analysis-progress'), timing=$('analysis-timing');
 const total=status.phase_total,done=status.phase_completed||0;
 bar.hidden=false;timing.hidden=false;
 if(status.phase==='complete'){bar.max=1;bar.value=1;}
 else if(status.overall_fraction!=null){bar.max=1;bar.value=status.overall_fraction;}else bar.removeAttribute('value');
 const format=seconds=>{const n=Math.max(0,Math.round(seconds));return `${Math.floor(n/3600)}h ${Math.floor(n%3600/60)}m ${n%60}s`;};
 const elapsed=(status.elapsed_seconds||0)+(job&&status.updated_at?Math.max(0,Date.now()/1000-status.updated_at):0);
 timing.textContent=`Elapsed ${format(elapsed)}`+(status.overall_fraction!=null?` · ${Math.round(status.overall_fraction*100)}% overall`: '')+(total?` · Current analysis: ${done}/${total} layers`: '')+
 (status.phase==='complete'?' · Complete':status.overall_estimated_remaining_seconds!=null?` · Rough remaining for all selected analyses and exports: ${format(status.overall_estimated_remaining_seconds)}`:' · Estimating remaining time…');
}
function render(data){
 $('view-label').textContent=data.run_id;$('figures').replaceChildren();
 $('availability').textContent='Layer and head effects, followed by downstream intervention results. Download the plotted data as CSV.';
 const groups=[['mechanistic','Component effects by layer'],['heads','Head heatmap'],['continuation','Downstream effects']];
 for(const [module,heading] of groups){
  if(module==='continuation')renderDetails(data);
  const section=document.createElement('section');section.className='analysis-section';
  const h=document.createElement('h2');h.textContent=heading;section.append(h);
  const figures=data.figures.filter(f=>new RegExp('(?:^|/)'+module+'_\\d+\\.png$').test(f.name));
  if(!figures.length){const p=document.createElement('p');p.textContent='No saved results. Select this analysis and create an analysis to produce them.';section.append(p);}
  for(const f of figures){
   const card=document.createElement('article');card.className='figure-card';
   const img=document.createElement('img');img.src=f.png;img.alt=heading+' · '+f.name;img.loading='lazy';
   const links=document.createElement('div');links.className='artifact-links';
   if(f.csv)links.append(link('Download plotted data CSV',f.csv));
   else links.textContent='Create an analysis to export CSV data for this older figure.';
   card.append(img,links);section.append(card);
  }
  $('figures').append(section);
 }
}
function renderDetails(data){
 // These fields add control/baseline information absent from the effect plots.
 const extraFields=['task','agent','round','model','layer','component','head','recipient','donor','control','baseline','donor_margin','recovery','position','anchor'];
 const tables=data.tables.filter(t=>['mechanistic','heads'].includes(t.name)&&t.rows.length&&t.header.some(h=>['baseline','recovery','control'].includes(h)));
 if(!tables.length)return;
 const section=document.createElement('section');section.className='analysis-section';
 const title=document.createElement('h2');title.textContent='Intervention details';section.append(title);
 const note=document.createElement('p');note.textContent='Baseline margins, controls and recovery details supplement the plotted effects. Recipient 0 = before peer delivery; 1 = after delivery. First 20 rows shown; CSV contains all fields and rows.';section.append(note);
 for(const t of tables){
  const card=document.createElement('section');card.className='table-card';
  const title=document.createElement('h3');title.textContent=t.name==='mechanistic'?'Component controls and baselines':'Head baselines';
  const columns=t.header.map((name,index)=>({name,index})).filter(c=>extraFields.includes(c.name));
  const scroll=document.createElement('div');scroll.className='table-scroll';const table=document.createElement('table');
  const head=document.createElement('thead');const hr=document.createElement('tr');
  for(const c of columns){const th=document.createElement('th');th.textContent=c.name;hr.append(th);}head.append(hr);
  const body=document.createElement('tbody');
  for(const row of t.rows){const tr=document.createElement('tr');for(const c of columns){const td=document.createElement('td');td.textContent=row[c.index];tr.append(td);}body.append(tr);}
  table.append(head,body);scroll.append(table);card.append(title,link('Download intervention details CSV',t.csv),scroll);section.append(card);
 }
 $('figures').append(section);
}

async function refreshAnalyses(){
 const data=await api('/api/analyses');const selected=$('saved-analysis').value;
 $('saved-analysis').replaceChildren(option('','Select a saved analysis…'),...data.analyses.map(r=>option(r.run_id,`${r.run_id} · ${r.phase}`)));
 $('saved-analysis').value=selected;activeRun=data.active_run;
 $('stop-analysis').disabled=!activeRun;$('stop-analysis').textContent=activeRun?`Stop ${activeRun}`:'Stop active run';
 return data;
}
async function loadAnalysis(name){
 const data=await api('/api/analysis?run_id='+encodeURIComponent(name));
 render(data);showProgress(data.status);$('status').textContent=data.status.phase||'Saved analysis';
 $('detail').textContent=data.status.error||`Loaded ${name}`;
 localStorage.setItem('opinion-lab-analysis',name);
}
$('saved-analysis').onchange=()=>{if($('saved-analysis').value)loadAnalysis($('saved-analysis').value).catch(error);};
$('stop-analysis').onclick=async()=>{try{
 $('stop-analysis').disabled=true;await api('/api/stop',{run_id:activeRun});$('status').textContent='Stopping…';
 $('detail').textContent='Waiting for the active process to stop. Saved outputs are retained.';
 }catch(e){error(e);$('stop-analysis').disabled=false;}};
function tasks(select,records){select.replaceChildren(...records.map(r=>option(r.task,`${r.task} · ${r.title} · ${r.config.schedule} · ${JSON.stringify(r.config.initialization)}`)));}
function selectionStatus(){if(job)return;const modules=[...document.querySelectorAll('[name=module]:checked')].map(e=>e.value);$('status').textContent='Ready';$('detail').textContent=modules.length?`Selected: ${modules.join(', ')}. All layers will be tested. Click Run selected analyses to start.`:'No new intervention selected. Select an analysis or re-export existing results.';$('analyze').textContent=modules.length?'Run selected analyses · all layers':'Re-export saved results';}
function events(){const record=sourceData?.records.find(r=>r.task===$('task').value);$('event').replaceChildren(...(record?.events||[]).map(r=>option(JSON.stringify(r),`${r.agent} · Round ${r.round} · ${r.model}`)));$('condition-summary').textContent=record?`${record.config.agents} agents · ${record.config.models.join(', ')} · ${record.config.argument_mode}`:'';$('analyze').disabled=!record||Boolean(job);selectionStatus();}
async function source(){sourceData=await api('/api/analysis?run_id='+encodeURIComponent($('source').value));tasks($('task'),sourceData.records);events();render(sourceData);}
document.querySelectorAll('[name=module]').forEach(e=>e.onchange=selectionStatus);
$('event').onchange=selectionStatus;
$('source').onchange=()=>source().catch(error);$('task').onchange=events;
$('comparison').onchange=async()=>{try{$('compare-task-label').hidden=!$('comparison').value;if($('comparison').value){const d=await api('/api/analysis?run_id='+encodeURIComponent($('comparison').value));tasks($('compare-task'),d.records);}}catch(e){error(e);}};
$('analyze').onclick=async()=>{try{
 $('error').textContent='';const modules=[...document.querySelectorAll('[name=module]:checked')].map(e=>e.value);const event=$('event').value?JSON.parse($('event').value):{};
 if(modules.length&&!event.agent)throw Error('This conversation has no eligible peer-delivery event. Choose another conversation.');
 $('analyze').disabled=true;$('status').textContent='Submitting analysis…';const result=await api('/api/analyze',{source_run:$('source').value,task:$('task').value,compare_run:$('comparison').value,compare_task:$('compare-task').value,modules,agent:event.agent,round:event.round,device:$('device').value,run_id:$('output-id').value});job=result.run_id;activeRun=job;$('stop-analysis').disabled=false;$('stop-analysis').textContent=`Stop ${job}`;localStorage.setItem('opinion-lab-analysis',job);showProgress({phase:'starting'});$('status').textContent='Starting analysis';$('detail').textContent=`Output: ${job}. Model analyses require new inference; saved conversations are reused.`;
 }catch(e){$('analyze').disabled=false;error(e);}};
setInterval(async()=>{if(polling)return;if(!job){await refreshAnalyses().catch(error);return;}polling=true;try{const d=await api('/api/analysis?run_id='+encodeURIComponent(job));showProgress(d.status);$('status').textContent=d.status.phase||'Starting';$('detail').textContent=d.status.error||`${job} · ${d.status.detail||'Working; results appear when analysis completes.'}`;if(['complete','failed','cancelled'].includes(d.status.phase)){render(d);$('detail').textContent=d.status.error||`Saved as ${job}`;if(d.status.error)error(Error(d.status.error));$('source').append(option(job,job));$('comparison').append(option(job,job));job=null;$('analyze').disabled=false;await refreshAnalyses();}}catch(e){error(e);}finally{polling=false;}},2000);
const themeKey='opinion-lab-theme';function theme(value){document.documentElement.dataset.theme=value;$('theme-toggle').textContent=value==='dark'?'Light theme':'Dark theme';localStorage.setItem(themeKey,value);}theme(localStorage.getItem(themeKey)||'light');$('theme-toggle').onclick=()=>theme(document.documentElement.dataset.theme==='dark'?'light':'dark');
(async()=>{try{const runs=await api('/api/runs');$('source').replaceChildren(...runs.map(r=>option(r,r)));$('comparison').append(...runs.map(r=>option(r,r)));if(runs.length)await source();else{$('analyze').disabled=true;$('status').textContent='No saved conversations yet';}
 const saved=await refreshAnalyses();const remembered=localStorage.getItem('opinion-lab-analysis');
 if(saved.active_run&&saved.analyses.some(r=>r.run_id===saved.active_run)){job=saved.active_run;$('analyze').disabled=true;await loadAnalysis(job);$('saved-analysis').value=job;}
 else if(remembered&&saved.analyses.some(r=>r.run_id===remembered)){await loadAnalysis(remembered);$('saved-analysis').value=remembered;}
 }catch(e){error(e);}})();
