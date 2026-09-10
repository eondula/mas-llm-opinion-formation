import unittest
from types import SimpleNamespace
import torch
from llm_mas.runtime import Runtime

class Encoding(dict):
    @property
    def input_ids(self):return self['input_ids']
    def to(self,device):return self

class Tokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        return ''.join(m['role']+':'+m['content']+'\n' for m in messages)+('assistant:' if add_generation_prompt else '')
    def __call__(self,text,return_tensors=None,return_offsets_mapping=False):
        ids=[ord(c)%32 for c in text]
        if return_tensors:return Encoding(input_ids=torch.tensor([ids]))
        return dict(input_ids=ids,offset_mapping=[(i,i+1) for i in range(len(text))])
    def convert_ids_to_tokens(self,ids):return [str(i) for i in ids]

class Attention(torch.nn.Module):
    def __init__(self):super().__init__();self.o_proj=torch.nn.Linear(4,4)
    def forward(self,x):return self.o_proj(x.cumsum(dim=1)/torch.arange(1,x.shape[1]+1)[None,:,None])
class Block(torch.nn.Module):
    def __init__(self):super().__init__();self.self_attn=Attention();self.mlp=torch.nn.Linear(4,4)
    def forward(self,x):return x+self.mlp(self.self_attn(x))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(32,4);self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([Block()]);self.output=torch.nn.Linear(4,32);self.config=SimpleNamespace(num_attention_heads=2,max_position_embeddings=10000)
    def forward(self,input_ids):
        x=self.embedding(input_ids)
        for block in self.model.layers:x=block(x)
        return SimpleNamespace(logits=self.output(x))
class Tiny(Runtime):
    def __init__(self):self.torch=torch;self.tokenizer=Tokenizer();self.model=Model().eval();self.device='cpu';self.name='toy'
    def activate(self):pass

class InterventionTests(unittest.TestCase):
    def test_actual_hooks(self):
        from llm_mas.interventions import sweep,head_sweep,source_span_sweep
        torch.manual_seed(1);runtime=Tiny();before=[{'role':'user','content':'Choose.'}];after=before+[{'role':'user','content':'Peer favors A.'}]
        scenario={'strategy_1':{'candidate':'A'},'strategy_2':{'candidate':'B'}};texts={'stance_probe':'{candidate_1} or {candidate_2}?'}
        rows=sweep(runtime,before,after,scenario,texts)
        self.assertEqual(len(rows),18)
        self.assertTrue(all(abs(r['delta'])<1e-6 for r in rows if r['control']=='self'))
        self.assertEqual(len(head_sweep(runtime,before,after,scenario,texts)),4)
        self.assertEqual(len(source_span_sweep(runtime,before,after,scenario,texts)),1)
        for module in runtime.model.modules():self.assertFalse(module._forward_hooks);self.assertFalse(module._forward_pre_hooks)
