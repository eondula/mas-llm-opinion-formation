"""Offline CPU check outside the collection set; never launches the full pilot."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from llm_mas.config import resolve,ROOT
from llm_mas.pilot import preset,runtime_options,shared_initialization
from llm_mas.runtime import Runtime
from llm_mas.prompts import templates
from llm_mas.stance_audit import audit
from llm_mas.engine import conversation
from llm_mas.storage import write
from llm_mas.tasks import question_task


def main():
    destination=ROOT/'results/preflight/pilot-v1'
    source=preset();models=list(source['model_revisions'])
    tasks=[question_task('preflight-arithmetic','Is two plus two equal to four?'),question_task('preflight-contradiction','Is two greater than ten?')]
    runtimes={name:Runtime(name,'cpu',**runtime_options(source,name)) for name in models}
    texts=templates()
    result=audit(destination,tasks,texts,runtimes,'preflight-only',lambda done,total:print(f'Audit {done}/{total}',flush=True))
    c=resolve(dict(mode='experiment',dataset=dict(path='data/strategyqa',split='train'),models=models,agents=2,agent_configuration='heterogeneous',rounds=1,repetitions=1,question_count=1,sentiment=False,message_similarity=False,device='cpu',max_argument_tokens=128,decoding='fixed_sampling',sampling_method='combined',sampling=dict(temperature=.7,top_p=.9,seed=42),paired_control=True,conditions=[dict(schedule='synchronous'),dict(schedule='no_peer')]))
    task=tasks[0];snapshot=shared_initialization(destination,c,task,texts,runtimes,'preflight-only')
    branches={s:conversation(dict(c,schedule=s),task,texts,runtimes,initial_rows=snapshot['rows']) for s in ['synchronous','no_peer']}
    assert branches['synchronous'][:2]==branches['no_peer'][:2]
    assert all(not r['deliveries'] for r in branches['no_peer'])
    write(destination/'matched_smoke.json',dict(status='passed',device='cpu',actual_generations=6,branches=branches))
    print('Preflight passed. CPU only; the full MPS audit runs before pilot collection.',flush=True)

if __name__=='__main__':main()
