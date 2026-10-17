/* Shared checkpoint catalog and model assignment controls. */
globalThis.OpinionModelControls = (() => {
 const available=['Qwen/Qwen2.5-1.5B-Instruct','Qwen/Qwen2.5-3B-Instruct','HuggingFaceTB/SmolLM2-1.7B-Instruct','TinyLlama/TinyLlama-1.1B-Chat-v1.0','allenai/OLMo-2-0425-1B-Instruct','HuggingFaceTB/SmolLM3-3B','Qwen/Qwen3-4B-Instruct-2507','microsoft/Phi-4-mini-instruct'];
 function create(doc=document){
  const get=id=>doc.getElementById(id),mode=get('agent-configuration'),shared=get('shared-model'),box=get('agent-models');
  let assignments=[];
  const option=value=>new Option(value,value);
  if(!shared.options.length)available.forEach(value=>shared.add(option(value)));
  function sync(){
   const mixed=mode.value==='heterogeneous';
   if(mixed)[...box.querySelectorAll('select')].forEach((node,i)=>assignments[i]=node.value);
   const models=mixed?assignments.slice(0,Number(get('agents').value)):[shared.value];
   get('models').value=models.join('\n');
   get('model-help').textContent=mixed?'Choose a checkpoint for each agent. Models may repeat across agents.':'All agents share this checkpoint and maintain separate histories.';
   if(models.includes('TinyLlama/TinyLlama-1.1B-Chat-v1.0'))get('model-help').textContent+=' TinyLlama has a short context window; use short conversations.';
   if(models.includes('allenai/OLMo-2-0425-1B-Instruct'))get('model-help').textContent+=' OLMo has a 4,096-token context window.';
  }
  function render(){
   const mixed=mode.value==='heterogeneous';
   const current=[...box.querySelectorAll('select')].map(node=>node.value);
   current.forEach((value,i)=>assignments[i]=value);
   get('shared-model-label').hidden=mixed;box.hidden=!mixed;box.replaceChildren();
   if(mixed)for(let i=0;i<Number(get('agents').value);i++){
    const label=doc.createElement('label');label.textContent='Agent '+(i<26?String.fromCharCode(65+i):i+1)+' model';
    const select=doc.createElement('select');available.forEach(value=>select.add(option(value)));
    select.value=assignments[i]||available[i%3];select.onchange=sync;
    label.append(select);box.append(label);
   }
   sync();
  }
  mode.onchange=render;shared.onchange=sync;render();
  return {render,sync};
 }
 return {available,create};
})();
