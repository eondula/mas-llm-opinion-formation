'use strict';
const toggle=document.getElementById('theme-toggle');
function setTheme(value){document.documentElement.dataset.theme=value;toggle.textContent=value==='dark'?'Light theme':'Dark theme';localStorage.setItem('opinion-lab-theme',value);}
setTheme(localStorage.getItem('opinion-lab-theme')||'light');
toggle.onclick=()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');
