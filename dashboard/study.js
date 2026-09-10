const $=id=>document.getElementById(id);let catalog=[],events=[],lastIndex=0;
async function api(path,data){const r=await fetch('/api/'+path,data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{});const v=await r.json();if(!r.ok)throw Error(v.error);return v;}
function config(){return {paper_suite:$('mode').value==='study'&&$('paper-suite').checked,run_id:$('run-id').value.trim()||undefined,mode:$('mode').value,agents:Number($('agents').value),models:$('models').value.split('\n').map(s=>s.trim()).filter(Boolean),scenarios:[...$('scenarios').selectedOptions].map(o=>o.value),rounds:Number($('rounds').value),initialization:$('initialization').value==='natural'?'natural':[...$('forced-strategies').querySelectorAll('select')].map(el=>Number(el.value)),schedule:$('schedule').value,argument_mode:$('argument_mode').value,max_argument_tokens:Number($('max_argument_tokens').value),research:$('mode').value==='study'?[...document.querySelectorAll('[name=research]:checked')].map(x=>x.value):[],conditions:$('mode').value==='study'&&$('conditions').value.trim()?JSON.parse($('conditions').value):[]};}
async function load(){catalog=await api('scenarios');$('scenarios').replaceChildren();for(const e of catalog){const o=new Option(e.data.title,e.path);o.selected=e.path.endsWith('/storage.json')||($('mode').value==='study'&&e.path.endsWith('/sensor.json'));$('scenarios').add(o);}}
function safe(fn){return async()=>{try{$('error').textContent='';await fn();}catch(e){$('error').textContent=e.message;}};}
$('mode').onchange=()=>{$('agents').disabled=$('mode').value==='preview';if($('agents').disabled)$('agents').value=2;$('scenarios').multiple=$('mode').value==='study';safe(load)();};$('mode').onchange();
function startingControls(){const box=$('forced-strategies'),previous=[...box.querySelectorAll('select')].map(el=>el.value);box.hidden=$('initialization').value!=='forced';box.replaceChildren();for(let i=0;i<Number($('agents').value);i++){const label=document.createElement('label');label.textContent=i<26?`Agent ${String.fromCharCode(65+i)}`:`Agent ${i+1}`;const select=document.createElement('select');select.id='forced-'+i;label.htmlFor=select.id;select.add(new Option('Strategy 1','1'));select.add(new Option('Strategy 2','2'));select.value=previous[i]||(i===0?'2':'1');box.append(label,select);}}
$('initialization').onchange=startingControls;$('agents').onchange=()=>{startingControls();draw(lastIndex);};const modeChange=$('mode').onchange;$('mode').onchange=()=>{modeChange();startingControls();};startingControls();
$('preview-prompts').onclick=safe(async()=>{$('prompt').textContent=JSON.stringify(await api('preview',config()),null,2);});
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
$('template').onclick=()=>{const e=catalog.find(e=>e.path===$('scenarios').value);if(e){const d=structuredClone(e.data);d.id='custom_scenario_001';$('scenario-editor').value=JSON.stringify(d,null,2);}};
$('save-scenario').onclick=safe(async()=>{await api('scenarios',JSON.parse($('scenario-editor').value));await load();});

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
 $('description').textContent=[settings.schedule?.replaceAll('_',' '),settings.initialization==='natural'?'Natural start':settings.initialization?'Forced start':null,settings.argument_mode?settings.argument_mode+' arguments':null].filter(Boolean).join(' · ');
}
function draw(index){
 updateHeading();
 const current=selectedTask||events[index]?.task, subset=events.slice(0,index+1).filter(e=>e.task===current), seen={};
 for(const e of subset)seen[e.row.agent]=e.row;
 const all=events.filter(e=>e.task===current);names.splice(0,names.length,...new Set(all.map(e=>e.row.agent)));names.sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
 const n=names.length||Number($('agents').value);if(!names.length)for(let i=0;i<n;i++)names.push(i<26?`Agent ${String.fromCharCode(65+i)}`:`Agent ${i+1}`);
 const arena=$('arena');arena.querySelectorAll('.agent').forEach(e=>e.remove());arena.classList.toggle('many-agents',n>2);arena.style.setProperty('--agent-scale',Number($('size').value)/100);
 if(n>2)arena.style.minHeight=Math.max(850,n*110)+'px';else arena.style.minHeight='';
 names.forEach((name,i)=>{const row=seen[name],node=agentTemplate.cloneNode(true);node.id='study-agent-'+i;node.className='agent '+(i%2?'right':'left');
 if(n>2){const angle=2*Math.PI*i/n-Math.PI/2;node.style.left=(50+33*Math.cos(angle))+'%';node.style.top=(50+34*Math.sin(angle))+'%';}
 node.querySelector('.speaker').textContent=name;node.querySelector('.name').textContent=name;node.querySelector('.message').textContent=row?.message||'';
 node.querySelector('.bubble').classList.toggle('has-response',Boolean(row));node.querySelector('.circle').className='circle '+(row?'s'+row.strategy:'unknown');
 node.querySelector('.strategy-label').textContent=row?'Strategy '+row.strategy:'No strategy yet';node.querySelector('.forced').textContent=row?.forced?'Forced start':row?'Stance-derived choice':'';
 node.querySelector('.activity').textContent=row?`Round ${row.round} · stance ${fmt(row.post.stance)}`:'Waiting';node.querySelector('.message').onclick=()=>{$('message').textContent=row?`${name} · ${row.model}\n${row.message}`:'';};
 if(n===2&&i===0)arena.insertBefore(node,arena.querySelector('.exchange'));else arena.append(node);
 });
 $('round').textContent=subset.length?'Round '+subset.at(-1).row.round:'Round —';$('checkpoint-label').textContent=$('round').textContent;
 $('delivery').textContent=subset.length?'Inspecting public responses':'Waiting for interaction';
 chart('stance-chart',all.length?Object.fromEntries(names.map(name=>[name,all.filter(e=>e.row.agent===name).map(e=>e.row.post.stance)])):null,'Post-response stance');
 const sent=all.some(e=>typeof e.row.sentiment==='number');chart('sentiment-chart',sent?Object.fromEntries(names.map(name=>[name,all.filter(e=>e.row.agent===name).map(e=>e.row.sentiment)])):null,'Sentiment available after scoring');
 const legend=document.querySelector('.legend');legend.replaceChildren();names.forEach((name,i)=>{const span=document.createElement('span');span.textContent=name;span.style.color=i%2?'var(--b)':'var(--a)';legend.append(span);});
 requestAnimationFrame(drawLinks);
}
let arrowFrame=null;
function drawLinks(){
 if(arrowFrame)cancelAnimationFrame(arrowFrame);
 const canvas=$('exchange-canvas'),rect=$('arena').getBoundingClientRect(),ratio=devicePixelRatio||1;
 canvas.width=rect.width*ratio;canvas.height=rect.height*ratio;
 const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
 const event=events[lastIndex];const deliveries=event?.task===selectedTask?(event.row.deliveries||[]):[];
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
 $('delivery').textContent=deliveries.length?deliveries.map(d=>`${d.speaker} → ${d.receiver}`).join(' · '):'No peer delivery at this step';
 const began=performance.now(),reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 function paint(now){ctx.clearRect(0,0,rect.width,rect.height);ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--exchange');ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2;
  for(const c of curves){ctx.beginPath();ctx.moveTo(c.sx,c.sy);ctx.quadraticCurveTo(c.cx,c.cy,c.ex,c.ey);ctx.stroke();ctx.beginPath();ctx.moveTo(c.ex,c.ey);ctx.lineTo(c.ex-10*Math.cos(c.end-.4),c.ey-10*Math.sin(c.end-.4));ctx.lineTo(c.ex-10*Math.cos(c.end+.4),c.ey-10*Math.sin(c.end+.4));ctx.closePath();ctx.fill();
   if(!reduced){const t=Math.min(1,(now-began)/1000),u=1-t;ctx.beginPath();ctx.arc(u*u*c.sx+2*u*t*c.cx+t*t*c.ex,u*u*c.sy+2*u*t*c.cy+t*t*c.ey,4,0,2*Math.PI);ctx.fill();}}
  if(!reduced&&now-began<1000)arrowFrame=requestAnimationFrame(paint);
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
  names.forEach((name,j)=>{const values=trajectories[name]||[],color=j?'var(--b)':'var(--a)';const points=values.map((v,i)=>`${x(i)},${y(v)}`).join(' ');make('polyline',{points,fill:'none',stroke:color,'stroke-width':2,'stroke-dasharray':j?'6 4':''});values.forEach((v,i)=>{const dot=make('circle',{cx:x(i),cy:y(v),r:3,fill:color});const title=document.createElementNS(ns,'title');title.textContent=`${name} · round ${i}: ${fmt(v)}`;dot.append(title);});});box.append(svg);
}

api('runs').then(runs=>{for(const run of runs)$('saved-run').add(new Option(run,run));});$('saved-run').onchange=()=>{currentRun=null;events=[];poll();};
