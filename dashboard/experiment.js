const $=id=>document.getElementById(id);
let experimentId='',busy=false,followLatest=true,pilotConfig=null,pilotReady=false,pilotRequested=false,pilotMessage='',newSetup=false,submitting=false,viewGeneration=0,pilotLoadId=0;
function lockPilot(){
 for(const node of $('settings').querySelectorAll('input,select,textarea'))node.disabled=false;
 samplingControls();compositionControls();
 if(pilotConfig)for(const node of $('settings').querySelectorAll('input,select,textarea'))node.disabled=node.id!=='experiment-id';
}
function updateControls(){
 const locked=busy||submitting;
 $('settings').disabled=locked;lockPilot();
 for(const id of ['pilot-device','pilot-preset','custom-preset'])$(id).disabled=locked;
 $('new-experiment').disabled=submitting;
 $('start').disabled=locked||(pilotRequested&&!pilotReady);
 $('launch-help').textContent=busy?'Another experiment is active. Follow the latest run to inspect or stop it.':submitting?'Starting experiment…':pilotRequested&&!pilotReady?(pilotMessage||'Loading pilot settings…'):pilotConfig?'Pilot ready. Its scientific settings are fixed; you can enter a new Experiment ID.':'Ready for a new experiment. Load the controlled pilot or configure a custom experiment.';
}
function newExperiment(){
 viewGeneration++;pilotLoadId++;newSetup=true;followLatest=false;experimentId='';
 pilotConfig=null;pilotReady=false;pilotRequested=false;pilotMessage='';
 $('saved').value='';$('experiment-id').value='';$('error').textContent='';$('run-error').textContent='';
 $('pilot-help').textContent='Choose Load controlled pilot for the full matrix, or configure a custom experiment.';
 renderSummary([]);updateControls();renderNewSetup();$('experiment-id').focus();
}
function renderNewSetup(){
 $('phase').textContent=busy?'New experiment setup · another run is active':'New experiment setup';
 $('progress').max=1;$('progress').value=0;$('timing').textContent='';$('output').textContent='';
 $('report-link').hidden=true;$('comparison-link').hidden=true;$('run-error').textContent='';$('stop').disabled=true;renderSummary([]);
}
$('new-experiment').onclick=newExperiment;
$('pilot-preset').onclick=async()=>{
 const requestId=++pilotLoadId;
 pilotRequested=true;pilotReady=false;pilotConfig=null;pilotMessage='Loading pilot settings…';$('error').textContent='';updateControls();
 try{
  const data=await api('pilot?device='+encodeURIComponent($('pilot-device').value));
  if(requestId!==pilotLoadId)return;
  if(!data.config)throw Error(data.error||'Pilot settings were not returned by the server.');
  pilotConfig=data.config;pilotReady=data.ready===true;pilotMessage=data.message||'Pilot is not ready. Freeze its protocol on this server.';
  $('pilot-device').value=data.config.device;const c=pilotConfig;
  $('composition-matrix').checked=true;$('model-q').value=c.model_configurations[0][0];$('model-s').value=c.model_configurations[1][0];
  for(const name of ['qq','ss','qs','sq'])$('composition-'+name).checked=true;
  for(const [id,value] of Object.entries({'runs':c.repetitions,'question-count':c.question_count,'agents':c.agents,'agent-configuration':c.agent_configuration,'rounds':c.rounds,'schedule':c.schedule,'initialization':'natural','partition':'train','split':'train','split-seed':42,'question-seed':42,'question-balance':'equal','decoding':c.decoding,'seed':c.seed,'sampling-method':c.sampling_method,'temperature':c.sampling.temperature,'top-p':c.sampling.top_p,'tokens':c.max_argument_tokens,'device':c.device,'question-ids':c.question_ids.join(', ')}))$(id).value=value;
  $('control').checked=true;$('sentiment').checked=true;$('message-similarity').checked=true;
  starts();modelPicker.render();for(const node of $('agent-models').querySelectorAll('select'))node.value=c.models[0];modelPicker.sync();
  $('pilot-help').textContent='Pilot '+c.device.toUpperCase()+': 10 balanced questions, QQ / SS / QS / SQ, 3 repetitions, 5 updates, natural starts, synchronous + shared-initialization no-peer. 120 pairs; 240 conversations; 2,640 responses with answer-selection and explanation stages. 512-token budget. '+(pilotReady?'Protocol '+data.protocol_hash:pilotMessage);
 }catch(error){if(requestId===pilotLoadId){pilotMessage=error.message;fail(error);}}
 finally{updateControls();}
};
$('custom-preset').onclick=()=>{
 pilotLoadId++;pilotConfig=null;pilotReady=false;pilotRequested=false;pilotMessage='';
 for(const [id,value] of Object.entries({'runs':3,'question-count':10,'agents':2,'rounds':5,'schedule':'synchronous','initialization':'natural','partition':'train','split-seed':42,'question-seed':42,'question-balance':'equal','decoding':'fixed_sampling','seed':42,'sampling-method':'combined','temperature':0.7,'top-p':0.9,'tokens':512,'question-ids':''}))$(id).value=value;
 $('split').replaceChildren(...[['train','Train'],['validation','Validation'],['test','Held-out test'],['all','All']].map(([value,label])=>new Option(label,value)));
 for(const option of $('questions').options)option.selected=false;
 $('composition-matrix').checked=true;for(const name of ['qq','ss','qs','sq'])$('composition-'+name).checked=true;
 $('model-q').value=OpinionModelControls.available[0];$('model-s').value=OpinionModelControls.available[2];
 $('control').checked=true;$('sentiment').checked=true;$('message-similarity').checked=true;starts();
 $('pilot-help').textContent='Editable pilot setup: QQ / SS / QS / SQ, 10 balanced questions, 3 repetitions, 5 updates, and matched no-peer controls. Change any setting below. This is a custom experiment, without a frozen pilot protocol.';
 updateControls();loadQuestions().catch(fail);
};
async function api(path,data){const response=await fetch('/api/'+path,data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{});const body=await response.json();if(!response.ok)throw Error(body.error||'Request failed');return body;}
function configuration(){
 if(pilotRequested&&!pilotReady)throw Error(pilotMessage||'Load a ready pilot before starting.');
 if(pilotConfig)return {...structuredClone(pilotConfig),run_id:$('experiment-id').value.trim()||undefined};
 let models=$('models').value.split('\n').map(v=>v.trim()).filter(Boolean),agents=Number($('agents').value),schedule=$('schedule').value;
 let model_configurations,agent_configuration=$('agent-configuration').value;
 if($('composition-matrix')?.checked){
  const q=$('model-q').value,s=$('model-s').value;
  if(q===s)throw Error('Choose different checkpoints for Model Q and Model S.');
  model_configurations=[['qq',[q,q]],['ss',[s,s]],['qs',[q,s]],['sq',[s,q]]].filter(([name])=>$('composition-'+name).checked).map(([,assignment])=>assignment);
  if(!model_configurations.length)throw Error('Select at least one agent composition.');
  models=model_configurations[0];agents=2;agent_configuration='heterogeneous';
 }
 const explicit=$('question-ids').value.split(/[\s,]+/).filter(Boolean);
 const ids=explicit.length?explicit:[...$('questions').selectedOptions].map(option=>option.value);
 return {mode:'experiment',output_layout:'experiment',run_id:$('experiment-id').value.trim()||undefined,
  models,agents,agent_configuration,...(model_configurations?{model_configurations}:{}),rounds:Number($('rounds').value),
  repetitions:Number($('runs').value),question_count:Number($('question-count').value),question_ids:ids,question_seed:Number($('question-seed').value),question_balance:$('question-balance')?.value||'random',
  dataset:{name:'strategyqa',partition:$('partition').value,split:$('split').value,split_seed:Number($('split-seed').value)},
  initialization:$('initialization').value==='natural'?'natural':[...$('forced-answers').querySelectorAll('select')].map(node=>Number(node.value)),
  schedule,paired_control:$('control').checked&&schedule!=='no_peer',conditions:$('control').checked&&schedule!=='no_peer'?[{schedule},{schedule:'no_peer'}]:[],
  decoding:$('decoding').value,sampling_method:$('sampling-method').value,seed:Number($('seed').value),
  sampling:{temperature:Number($('temperature').value),top_p:Number($('top-p').value),seed:Number($('seed').value)},
  max_argument_tokens:Number($('tokens').value),message_similarity:$('message-similarity').checked,device:$('device').value,sentiment:$('sentiment').checked};
}
function compositionControls(){
 const matrix=$('composition-matrix').checked;
 $('composition-settings').hidden=!matrix;$('single-model-settings').hidden=matrix;
 for(const node of $('single-model-settings').querySelectorAll('input,select,textarea'))node.disabled=matrix;
 if(matrix&&$('agents').value!=='2'){$('agents').value=2;starts();}
}
function starts(){const box=$('forced-answers'),old=[...box.querySelectorAll('select')].map(n=>n.value);box.replaceChildren();box.hidden=$('initialization').value!=='forced';for(let i=0;i<Number($('agents').value);i++){const label=document.createElement('label');label.textContent=i<26?'Agent '+String.fromCharCode(65+i):'Agent '+(i+1);const select=document.createElement('select');select.add(new Option('Yes','1'));select.add(new Option('No','2'));select.value=old[i]||(i%2?'2':'1');label.append(select);box.append(label);}}
const percent=v=>typeof v==='number'?(v*100).toFixed(1)+'%':'—';
const decimal=v=>typeof v==='number'?v.toFixed(4):'—';
function renderSummary(rows){$('summary-section').hidden=!rows?.length;const body=$('summary').querySelector('tbody');body.replaceChildren();for(const row of rows||[]){const tr=document.createElement('tr');const cells=[row.models,row.agents,row.schedule,row.initialization==='"natural"'?'Natural':row.initialization,row.research_split,row.runs,row.questions,row.conversations,percent(row.initial_accuracy),percent(row.final_accuracy),percent(row.initial_disagreement_rate),percent(row.final_consensus_rate),percent(row.final_majority_accuracy),decimal(row.mean_absolute_stance_change),row.answer_reversals,row.correct_to_incorrect,row.incorrect_to_correct,decimal(row.mean_final_dispersion),decimal(row.mean_adjacent_message_cosine),row.invalid_responses];for(const value of cells){const td=document.createElement('td');td.textContent=value??'—';tr.append(td);}body.append(tr);}}
async function saved(){const rows=await api('experiments');const select=$('saved'),previous=select.value;select.replaceChildren(new Option('Select an experiment',''),...rows.map(r=>new Option(r.id+' · '+r.phase,r.id)));select.value=experimentId||previous;}
async function poll(){
 const generation=viewGeneration;
 const data=await api('experiment'+(!newSetup&&!followLatest&&experimentId?'?run_id='+encodeURIComponent(experimentId):''));
 if(generation!==viewGeneration)return;
 busy=Boolean(data.worker_busy);updateControls();
 if(newSetup){renderNewSetup();return;}
 if((followLatest||!experimentId)&&data.run_id)experimentId=data.run_id;
 if(followLatest&&experimentId)$('saved').value=experimentId;
 const status=data.status;$('stop').disabled=!data.can_stop;
 const completed=status.completed||0,total=status.total||0;
 $('phase').textContent=(data.run_id?data.run_id+' · ':'')+status.phase.replaceAll('_',' ')+(total?' · '+completed+' / '+total+' responses':'');
 if(data.remote_progress)$('phase').textContent+=' · last saved progress from '+data.worker_host+' (open the dashboard there for live control)';
 const phaseTotal=status.phase_total,phaseDone=status.phase_completed||0;
 if(phaseTotal&&status.phase!=='conversation'&&status.phase!=='complete')$('phase').textContent+=' · phase '+phaseDone+' / '+phaseTotal;
 $('progress').max=(status.phase!=='conversation'&&status.phase!=='complete'&&phaseTotal)||total||1;
 $('progress').value=status.phase==='complete'?total||1:(status.phase!=='conversation'&&phaseTotal?phaseDone:completed);
 $('timing').textContent=status.elapsed_seconds!=null?'Elapsed '+Math.round(status.elapsed_seconds)+'s'+(data.running&&status.estimated_remaining_seconds!=null?' · estimated simulation time remaining '+Math.round(status.estimated_remaining_seconds)+'s':''):'';
 if(status.generation_total)$('timing').textContent+=' · generated '+(status.generated_responses||0)+' / '+status.generation_total;
 $('output').textContent=data.output_directory?'Results: '+data.output_directory:'';
 $('report-link').hidden=!data.report_available;$('report-link').href='/results/'+encodeURIComponent(data.run_id||'')+'/index.html';$('comparison-link').hidden=!data.report_available;
 $('run-error').textContent=status.error?'Saved run '+(data.run_id||'')+': '+status.error+' Use New experiment to set up a fresh run.':'';
 renderSummary(status.phase==='complete'?data.summary:[]);
 if(['complete','failed','cancelled'].includes(status.phase))await saved();
}
function fail(error){$('error').textContent=error.message;}
$('pilot-device').onchange=()=>$('pilot-preset').click();
$('configuration').onsubmit=async event=>{
 event.preventDefault();$('error').textContent='';
 if(busy||submitting)return;
 if(pilotRequested&&!pilotReady){fail(Error(pilotMessage||'Freeze this device protocol on the server before starting.'));return;}
 submitting=true;updateControls();renderSummary([]);
 try{
  const result=await api('experiment',configuration());
  viewGeneration++;experimentId=result.run_id;newSetup=false;followLatest=true;$('run-error').textContent='';
  localStorage.setItem('opinion-experiment',experimentId);$('phase').textContent='Starting '+experimentId;
  await saved();await poll();
 }catch(error){fail(error);}
 finally{submitting=false;updateControls();}
};
$('stop').onclick=async()=>{try{await api('stop',{run_id:experimentId});$('phase').textContent='Stopping…';}catch(error){fail(error);}};
$('open').onclick=async()=>{viewGeneration++;newSetup=false;followLatest=false;experimentId=$('saved').value;localStorage.setItem('opinion-experiment',experimentId);$('error').textContent='';try{await poll();}catch(error){fail(error);}};
$('latest').onclick=async()=>{viewGeneration++;newSetup=false;followLatest=true;experimentId='';$('error').textContent='';try{await poll();await saved();}catch(error){fail(error);}};
const modelPicker=OpinionModelControls.create();
for(const id of ['model-q','model-s'])for(const value of OpinionModelControls.available)$(id).add(new Option(value,value));
$('model-s').value=OpinionModelControls.available[2];
$('composition-matrix').onchange=compositionControls;compositionControls();
$('initialization').onchange=starts;$('agents').onchange=()=>{starts();modelPicker.render();};starts();
async function loadQuestions(){
 const box=$('questions'),selected=new Set([...box.selectedOptions].map(option=>option.value));
 const data=await api('questions?partition='+encodeURIComponent($('partition').value)+'&split='+encodeURIComponent($('split').value)+'&split_seed='+encodeURIComponent($('split-seed').value));
 box.replaceChildren(...data.questions.map(question=>{const option=new Option(question.question+' · '+question.id,question.id);option.selected=selected.has(question.id);return option;}));
 $('dataset-help').textContent=data.questions.length+' questions in this split. Leave selection empty to sample the configured count. '+(data.labeled?'Labels are used after simulation only.':'Correctness is unavailable for the official test partition.');
}
$('partition').onchange=()=>{$('split').replaceChildren(...($('partition').value==='test'?[['official_test','Official test · unlabeled']]:[['train','Train'],['validation','Validation'],['test','Held-out test'],['all','All']]).map(([value,label])=>new Option(label,value)));loadQuestions().catch(fail);};
$('split').onchange=()=>loadQuestions().catch(fail);$('split-seed').onchange=()=>loadQuestions().catch(fail);
function samplingControls(){
 const sampled=$('decoding').value==='fixed_sampling',method=$('sampling-method').value;
 $('sampling-controls').hidden=!sampled;
 $('temperature-label').hidden=method==='top_p';$('temperature').disabled=!sampled||method==='top_p';
 $('top-p-label').hidden=method==='temperature';$('top-p').disabled=!sampled||method==='temperature';
 $('sampling-help').textContent=method==='temperature'?'Top-p is fixed at 1; top-k is disabled.':method==='top_p'?'Temperature is fixed at 1; top-k is disabled.':'Both controls apply; top-k is disabled.';
}
$('decoding').onchange=samplingControls;$('sampling-method').onchange=samplingControls;samplingControls();
$('preview-prompts').onclick=async()=>{const dialog=$('prompt-dialog');$('prompt').textContent='Loading prompts…';dialog.showModal();try{$('prompt').textContent=JSON.stringify(await api('preview',configuration()),null,2);}catch(error){$('prompt').textContent=error.message;fail(error);}};
$('close-prompt').onclick=()=>$('prompt-dialog').close();

(async()=>{try{await Promise.all([saved(),loadQuestions()]);await poll();}catch(error){fail(error);}setInterval(()=>poll().catch(fail),2000);})();
