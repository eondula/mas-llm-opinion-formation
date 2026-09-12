const $=id=>document.getElementById(id);let catalog=[],events=[],lastIndex=0;
async function api(path,data){const r=await fetch('/api/'+path,data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{});const v=await r.json();if(!r.ok)throw Error(v.error);return v;}
function config(){return {paper_suite:$('mode').value==='study'&&$('paper-suite').checked,run_id:$('run-id').value.trim()||undefined,mode:$('mode').value,agents:Number($('agents').value),agent_configuration:$('agent-configuration').value,models:$('models').value.split('\n').map(s=>s.trim()).filter(Boolean),scenarios:[...$('scenarios').selectedOptions].map(o=>o.value),rounds:Number($('rounds').value),initialization:$('initialization').value==='natural'?'natural':[...$('forced-strategies').querySelectorAll('select')].map(el=>Number(el.value)),schedule:$('schedule').value,argument_mode:$('argument_mode').value,max_argument_tokens:Number($('max_argument_tokens').value),research:$('mode').value==='study'?[...document.querySelectorAll('[name=research]:checked')].map(x=>x.value):[],conditions:$('mode').value==='study'&&$('conditions').value.trim()?JSON.parse($('conditions').value):[]};}
async function load(){catalog=await api('scenarios');$('scenarios').replaceChildren();for(const e of catalog){const o=new Option(e.data.title,e.path);o.selected=e.path.endsWith('/storage.json')||($('mode').value!=='preview'&&e.path.endsWith('/sensor.json'));$('scenarios').add(o);}}
function safe(fn){return async()=>{try{$('error').textContent='';await fn();}catch(e){$('error').textContent=e.message;}};}
$('mode').onchange=()=>{$('agents').disabled=false;$('scenarios').multiple=$('mode').value!=='preview';safe(load)();};$('mode').onchange();
function startingControls(){const box=$('forced-strategies'),previous=[...box.querySelectorAll('select')].map(el=>el.value);box.hidden=$('initialization').value!=='forced';box.replaceChildren();for(let i=0;i<Number($('agents').value);i++){const label=document.createElement('label');label.textContent=i<26?`Agent ${String.fromCharCode(65+i)}`:`Agent ${i+1}`;const select=document.createElement('select');select.id='forced-'+i;label.htmlFor=select.id;select.add(new Option('Strategy 1','1'));select.add(new Option('Strategy 2','2'));select.value=previous[i]||(i===0?'2':'1');box.append(label,select);}}
$('initialization').onchange=startingControls;$('agents').onchange=()=>{startingControls();modelControls();draw(lastIndex);};const modeChange=$('mode').onchange;$('mode').onchange=()=>{modeChange();startingControls();};startingControls();
$('preview-prompts').onclick=safe(async()=>{const dialog=$('prompt-dialog');$('prompt').textContent='Loading configuration and prompts…';dialog.showModal();try{const result=await api('preview',config());$('prompt').textContent=JSON.stringify(result,null,2);}catch(error){$('prompt').textContent=error.message;throw error;}});
$('run').onclick=safe(async()=>{
 $('run').disabled=true;$('run-state').textContent='Starting';
 try {
  const result=await api('run',config());
  $('saved-run').value='';following=true;currentRun=null;selectedTask=null;events=[];
  $('title').textContent='Preparing the conversation…';
  $('description').textContent='Run '+result.run_id+' · loading the model';
  await poll();
 } catch(error){$('run').disabled=false;$('run-state').textContent='Unable to start';throw error;}
});$('stop').onclick=safe(()=>api('stop',{}));
$('close-prompt').onclick=()=>$('prompt-dialog').close();
$('new-scenario').onclick=()=>{$('scenario-error').textContent='';$('scenario-dialog').showModal();};
$('close-scenario').onclick=()=>$('scenario-dialog').close();
$('scenario-template').onclick=()=>{
 const data=catalog.find(e=>e.path===$('scenarios').value)?.data;if(!data)return;
 for(const input of $('scenario-form').querySelectorAll('[name]')){
  const [key,field]=input.name.split('.');const value=field?data[key][field]:data[key];
  input.value=key==='id'?'custom_'+Date.now():Array.isArray(value)?value.join('\n'):value;
 }
};
$('scenario-form').onsubmit=async event=>{
 event.preventDefault();const button=event.submitter;button.disabled=true;$('scenario-error').textContent='';
 try{
  const data={strategy_1:{},strategy_2:{}};
  for(const input of $('scenario-form').querySelectorAll('[name]')){
   const [key,field]=input.name.split('.');const value=input.value.trim();
   if(field)data[key][field]=field==='arguments'?value.split('\n').map(x=>x.trim()).filter(Boolean):value;else data[key]=value;
  }
  const result=await api('scenarios',data);await load();
  for(const option of $('scenarios').options)option.selected=option.value===result.saved;
  $('scenario-dialog').close();$('error').textContent='';
 }catch(error){$('scenario-error').textContent=error.message;}finally{button.disabled=false;}
};

let selectedTask=null, following=true, timer=null, currentRun=null, activeConfig=null, activeTasks=[], lastPhase=null;
const names=[];
const fmt=n=>typeof n==='number'?n.toFixed(3):'—';
$('theme-toggle').onclick=()=>{const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';$('theme-toggle').textContent=dark?'Light theme':'Dark theme';localStorage.setItem('opinion-lab-theme',dark?'dark':'light');draw(lastIndex);};
document.documentElement.dataset.theme=localStorage.getItem('opinion-lab-theme')||'light';
$('theme-toggle').textContent=document.documentElement.dataset.theme==='dark'?'Light theme':'Dark theme';
const agentTemplate=$('agent-a').cloneNode(true);
function updateHeading(){
 const task=selectedTask||events[lastIndex]?.task;
 if(!task)return;
 const [conditionIndex,scenarioIndex]=task.split('_').map(Number);
 const scenario=activeConfig?.scenarios?.[scenarioIndex];
 const settings=activeTasks[conditionIndex]||{...activeConfig,...activeConfig?.conditions?.[conditionIndex]};
 $('title').textContent=scenario?.title||'Conversation in progress';
 $('description').textContent=[settings.schedule?.replaceAll('_',' '),settings.initialization==='natural'?'Natural start':settings.initialization?'Forced start':null,settings.argument_mode?({predefined:'Predefined (cyclic)',agent_selected:'Conversational (fixed list)',combined_reasoning:'Combined prompt (flexible reasoning)',combined_fixed:'Combined prompt (fixed list)',generated:'Generated',conversational:'Conversational (generated)'}[settings.argument_mode]||settings.argument_mode)+' arguments':null].filter(Boolean).join(' · ');
}
function draw(index){
 updateHeading();
 const current=selectedTask||events[index]?.task, subset=events.slice(0,index+1).filter(e=>e.task===current), seen={};
 for(const e of subset)seen[e.row.agent]=e.row;
 const all=events.filter(e=>e.task===current);
 const taskIndex=Number((current||'0').split('_')[0]);
 const settings=activeTasks[taskIndex]||activeConfig;
 const n=settings?.agents||Number($('agents').value);
 names.splice(0,names.length,...Array.from({length:n},(_,i)=>i<26?`Agent ${String.fromCharCode(65+i)}`:`Agent ${i+1}`));
 const visible=responseFrame(events,index,current,settings||{agents:n,schedule:$('schedule').value});
 const checkpoints=[...new Set(settings?.models||$('models').value.split('\n').filter(Boolean))];
 const arena=$('arena');arena.querySelectorAll('.agent').forEach(e=>e.remove());arena.classList.toggle('many-agents',n>2);arena.style.setProperty('--agent-scale',Number($('size').value)/100);
 if(n>2)arena.style.minHeight=Math.max(850,n*110)+'px';else arena.style.minHeight='';
 names.forEach((name,i)=>{const row=seen[name],node=agentTemplate.cloneNode(true);node.id='study-agent-'+i;const checkpoint=row?.model||settings?.models?.[settings.models.length===1?0:i]||checkpoints[0];node.style.setProperty('--agent-color',['var(--checkpoint-1)','var(--checkpoint-2)','var(--checkpoint-3)'][Math.max(0,checkpoints.indexOf(checkpoint))%3]);node.className='agent '+(i%2?'right':'left');
 if(n>2){const angle=2*Math.PI*i/n-Math.PI/2;node.style.left=(50+33*Math.cos(angle))+'%';node.style.top=(50+34*Math.sin(angle))+'%';}
 node.querySelector('.speaker').textContent=name;node.querySelector('.name').textContent=name;node.querySelector('.message').textContent=row?.message||'';
 const shown=visible.find(r=>r.agent===name);node.classList.toggle('is-active',Boolean(shown));node.querySelector('.message').textContent=shown?.message||'';
 node.querySelector('.bubble').classList.toggle('has-response',Boolean(shown));node.querySelector('.circle').className='circle '+(row?'s'+row.strategy:'unknown');
 node.querySelector('.strategy-label').textContent=row?'Strategy '+row.strategy:'No strategy yet';node.querySelector('.forced').textContent=row?.forced?'Forced start':row?'Stance-derived choice':'';
 node.querySelector('.activity').textContent=row?`Round ${row.round} · stance ${fmt(row.post.stance)}`:'Waiting';node.querySelector('.message').title=row?.model||'';
 if(n===2&&i===0)arena.insertBefore(node,arena.querySelector('.exchange'));else arena.append(node);
 });
 $('round').textContent=subset.length?'Round '+subset.at(-1).row.round:'Round —';$('checkpoint-label').textContent=$('round').textContent;
 $('delivery').textContent='';$('delivery').hidden=true;
 chart('stance-chart',all.length?Object.fromEntries(names.map(name=>[name,all.filter(e=>e.row.agent===name).map(e=>e.row.post.stance)])):null,'Post-response stance');
 const sent=all.some(e=>typeof e.row.sentiment==='number');chart('sentiment-chart',sent?Object.fromEntries(names.map(name=>[name,all.filter(e=>e.row.agent===name).map(e=>e.row.sentiment)])):null,'Sentiment available after scoring');
 const legend=document.querySelector('.legend');legend.replaceChildren();names.forEach((name,i)=>{const span=document.createElement('span');span.textContent=name;span.style.color=['var(--plot-1)','var(--plot-2)','var(--plot-3)','var(--plot-4)'][i%4];legend.append(span);});
 requestAnimationFrame(drawLinks);
}
let arrowFrame=null,lastAnimatedDelivery=null;
function drawLinks(){
 if(arrowFrame)cancelAnimationFrame(arrowFrame);
 const canvas=$('exchange-canvas'),rect=$('arena').getBoundingClientRect(),ratio=devicePixelRatio||1;
 canvas.width=rect.width*ratio;canvas.height=rect.height*ratio;
 const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
 const event=events[lastIndex];
 const settings=activeTasks[Number((selectedTask||'0').split('_')[0])]||activeConfig||{};
 const deliveries=deliveryFrame(events,lastIndex,selectedTask,settings); 
 const curves=[];
 for(const delivery of deliveries){
  const from=names.indexOf(delivery.speaker),to=names.indexOf(delivery.receiver);
  const source=$('study-agent-'+from)?.querySelector('.circle'),target=$('study-agent-'+to)?.querySelector('.circle');if(!source||!target)continue;
  const a=source.getBoundingClientRect(),b=target.getBoundingClientRect();
  const x=a.x+a.width/2-rect.x,y=a.y+a.height/2-rect.y,xx=b.x+b.width/2-rect.x,yy=b.y+b.height/2-rect.y;
  const dx=xx-x,dy=yy-y,d=Math.hypot(dx,dy)||1,bend=Math.min(90,d*.25),cx=(x+xx)/2-dy/d*bend,cy=(y+yy)/2+dx/d*bend;
  const start=Math.atan2(cy-y,cx-x),end=Math.atan2(yy-cy,xx-cx);
  curves.push({sx:x+Math.cos(start)*(a.width/2+4),sy:y+Math.sin(start)*(a.height/2+4),cx,cy,ex:xx-Math.cos(end)*(b.width/2+7),ey:yy-Math.sin(end)*(b.height/2+7),end});
 }
 $('delivery').textContent='';$('delivery').hidden=true;
 const deliveryKey=JSON.stringify([currentRun,selectedTask,lastIndex,deliveries]);
 const animate=deliveryKey!==lastAnimatedDelivery&&!['complete','failed','cancelled'].includes(lastPhase);lastAnimatedDelivery=deliveryKey;
 const began=performance.now(),reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 function paint(now){ctx.clearRect(0,0,rect.width,rect.height);ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--exchange');ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2;
  for(const c of curves){ctx.beginPath();ctx.moveTo(c.sx,c.sy);ctx.quadraticCurveTo(c.cx,c.cy,c.ex,c.ey);ctx.stroke();ctx.beginPath();ctx.moveTo(c.ex,c.ey);ctx.lineTo(c.ex-10*Math.cos(c.end-.4),c.ey-10*Math.sin(c.end-.4));ctx.lineTo(c.ex-10*Math.cos(c.end+.4),c.ey-10*Math.sin(c.end+.4));ctx.closePath();ctx.fill();
   if(!reduced&&animate&&now-began<1400){const t=Math.min(1,(now-began)/1400),u=1-t;ctx.beginPath();ctx.arc(u*u*c.sx+2*u*t*c.cx+t*t*c.ex,u*u*c.sy+2*u*t*c.cy+t*t*c.ey,4,0,2*Math.PI);ctx.fill();}}
  if(!reduced&&animate&&curves.length&&now-began<1400)arrowFrame=requestAnimationFrame(paint);else arrowFrame=null;
 }paint(began);
}
$('size').oninput=()=>draw(lastIndex);window.addEventListener('resize',drawLinks);
$('checkpoint').oninput=()=>{following=false;lastIndex=Number($('checkpoint').value);draw(lastIndex);};
$('follow').onclick=()=>{following=true;$('follow').textContent='Following live';lastIndex=Math.max(0,events.length-1);selectedTask=events[lastIndex]?.task;draw(lastIndex);};
$('condition').onchange=()=>{selectedTask=$('condition').value;following=false;lastIndex=events.map(e=>e.task).lastIndexOf(selectedTask);draw(lastIndex);};
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play responses';return;}following=false;const indices=events.map((e,i)=>e.task===selectedTask?i:-1).filter(i=>i>=0);let at=0;timer=setInterval(()=>{if(at>=indices.length){clearInterval(timer);timer=null;$('play').textContent='Play responses';return;}lastIndex=indices[at++];$('checkpoint').value=lastIndex;draw(lastIndex);},Number($('playback-speed').value));$('play').textContent='Pause';};
async function poll(){try{const d=await api('status'+($('saved-run').value?'?run_id='+encodeURIComponent($('saved-run').value):''));activeConfig=d.config;activeTasks=d.tasks||[];$('zip-download').hidden=!d.run_id;$('zip-download').href='/results/'+encodeURIComponent(d.run_id||'')+'/paper_outputs.zip';$('figure-download').hidden=!d.run_id;$('figure-download').href='/results/'+encodeURIComponent(d.run_id||'')+'/figures/index.html';updateHeading();$('connection').textContent='Connected';$('run-state').textContent=d.status.phase;$('progress').textContent=`${d.status.completed||0} / ${d.status.total||0}`;$('study-progress').max=d.status.total||1;$('study-progress').value=d.status.completed||0;$('timing').textContent=`Elapsed ${Math.round(d.status.elapsed_seconds||0)}s · ETA ${d.status.estimated_remaining_seconds==null?'estimating':Math.round(d.status.estimated_remaining_seconds)+'s'}`;$('run').disabled=d.running;$('stop').disabled=!d.running;
 if(d.run_id!==currentRun||d.events.length!==events.length||lastPhase!==d.status.phase){lastPhase=d.status.phase;currentRun=d.run_id;events=d.events;const tasks=[...new Set(events.map(e=>e.task))];$('condition').replaceChildren(...tasks.map(t=>new Option('Conversation '+t,t)));if(following||!tasks.includes(selectedTask)){lastIndex=Math.max(0,events.length-1);selectedTask=events[lastIndex]?.task;} $('condition').value=selectedTask||'';$('checkpoint').max=Math.max(0,events.length-1);$('checkpoint').value=lastIndex;$('checkpoint').disabled=!events.length;$('play').disabled=!events.length;draw(lastIndex);}if(d.status.error)$('error').textContent=d.status.error;
 }catch(e){$('connection').textContent='Disconnected';$('error').textContent=e.message;}}
setInterval(poll,2000);poll();draw(0);
function chart(container, trajectories, caption) {
  const box=$(container);box.replaceChildren();
  if(!trajectories) {const p=document.createElement('p');p.textContent=caption;box.append(p);return;}
  const ns='http://www.w3.org/2000/svg', W=520,H=235,l=42,r=15,t=20,b=32;
  const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('role','img');svg.setAttribute('aria-label',caption);
  const make=(tag,attrs,text)=>{const el=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(text!==undefined)el.textContent=text;svg.append(el);return el;};
  const n=Math.max(...names.map(k=>trajectories[k]?.length||0));const x=i=>l+i/Math.max(n-1,1)*(W-l-r),y=v=>t+(1-v)/2*(H-t-b);
  for(const v of [-1,0,1]){make('line',{x1:l,y1:y(v),x2:W-r,y2:y(v),stroke:'var(--edge)'});make('text',{x:l-9,y:y(v)+4,'text-anchor':'end','font-size':11,fill:'var(--muted)'},v);}
  const step=Math.max(1,Math.ceil((n-1)/8));for(let i=0;i<n;i++)if(i%step===0||i===n-1)make('text',{x:x(i),y:H-12,'text-anchor':'middle','font-size':11,fill:'var(--muted)'},i);
  names.forEach((name,j)=>{const values=trajectories[name]||[],color=['var(--plot-1)','var(--plot-2)','var(--plot-3)','var(--plot-4)'][j%4];const points=values.map((v,i)=>`${x(i)},${y(v)}`).join(' ');make('polyline',{points,fill:'none',stroke:color,'stroke-width':2,'stroke-dasharray':['','6 4','2 3','9 3 2 3'][j%4]});values.forEach((v,i)=>{const dot=make('circle',{cx:x(i),cy:y(v),r:3,fill:color});const title=document.createElementNS(ns,'title');title.textContent=`${name} · round ${i}: ${fmt(v)}`;dot.append(title);});});box.append(svg);
}

api('runs').then(runs=>{for(const run of runs)$('saved-run').add(new Option(run,run));});$('saved-run').onchange=()=>{currentRun=null;events=[];poll();};

// Synchronous delivery is displayed as one complete round; sequential delivery
// follows the selected receiver's actual recorded event.
function deliveryFrame(records,index,task,settings){
 const event=records[index];
 if(!event||event.task!==task||settings.schedule==='no_peer')return [];
 if(settings.schedule!=='synchronous')return event.row.deliveries||[];
 const round=records.slice(0,index+1).filter(e=>e.task===task&&e.row.round===event.row.round);
 if(new Set(round.map(e=>e.row.agent)).size<settings.agents)return [];
 return round.flatMap(e=>e.row.deliveries||[]);
}

function responseFrame(records,index,task,settings){
 const event=records[index];if(!event||event.task!==task)return [];
 if(settings.schedule!=='synchronous')return [event.row];
 const rows=records.slice(0,index+1).filter(e=>e.task===task&&e.row.round===event.row.round).map(e=>e.row);
 return new Set(rows.map(r=>r.agent)).size===settings.agents?rows:[];
}

function syncModels(){
 const mixed=$('agent-configuration').value==='heterogeneous';
 const models=mixed?[...$('agent-models').querySelectorAll('select')].map(s=>s.value):[$('shared-model').value];
 $('models').value=models.join('\n');
 $('model-help').textContent=mixed?'Choose any available model for each agent. Models may repeat across agents, including groups of four or more.':'All agents share this checkpoint and maintain separate histories.';
}
function modelControls(){
 const available=['Qwen/Qwen2.5-1.5B-Instruct','Qwen/Qwen2.5-3B-Instruct','HuggingFaceTB/SmolLM2-1.7B-Instruct'];
 const mixed=$('agent-configuration').value==='heterogeneous',box=$('agent-models');
 const previous=[...box.querySelectorAll('select')].map(s=>s.value);
 if(!$('shared-model').options.length)available.forEach(m=>$('shared-model').add(new Option(m,m)));
 $('shared-model-label').hidden=mixed;box.hidden=!mixed;box.replaceChildren();
 for(let i=0;mixed&&i<Number($('agents').value);i++){
  const label=document.createElement('label');label.textContent='Agent '+(i<26?String.fromCharCode(65+i):i+1)+' model';
  const select=document.createElement('select');available.forEach(m=>select.add(new Option(m,m)));select.value=previous[i]||available[i%3];select.onchange=syncModels;label.append(select);box.append(label);
 }
 syncModels();
}
$('agent-configuration').onchange=modelControls;
$('shared-model').onchange=syncModels;
modelControls();

// Experiment C fixes the comparison factors; other modes keep their controls.
const previousExperimentModeChange = $('mode').onchange;
$('mode').onchange = () => {
 previousExperimentModeChange();
 const fixed = $('mode').value === 'experiment_c';
 $('experiment-c-help').hidden = !fixed;
 for (const id of ['agents','agent-configuration','shared-model','initialization','schedule','argument_mode','paper-suite','conditions']) $(id).disabled = fixed;
 for (const el of document.querySelectorAll('#agent-models select, #forced-strategies select, [name=research]')) el.disabled = fixed;
};
