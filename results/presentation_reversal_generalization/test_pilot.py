import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
spec=importlib.util.spec_from_file_location('sweep',Path(__file__).with_name('run.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Inputs(dict):
    @property
    def input_ids(self):return self['input_ids']
class Tests(unittest.TestCase):
    def test_float32_likelihood_matches_sum(self):
        r=m.FloatRuntime.__new__(m.FloatRuntime);r.torch=torch
        r.inputs=lambda messages,candidate:('P',Inputs(input_ids=torch.tensor([[0,1,2]])))
        r.encode_rendered=lambda text:dict(input_ids=[0])
        logits=torch.arange(12,dtype=torch.float32).reshape(1,3,4)
        r.model=lambda **kwargs:SimpleNamespace(logits=logits)
        score,values,_=r.likelihood([], 'AB')
        expected=torch.log_softmax(logits[:,:2,:],-1).gather(-1,torch.tensor([[[1],[2]]])).flatten()
        self.assertEqual(score,expected.sum(dtype=torch.float32).item());self.assertEqual(values,expected.tolist())
        r.model=lambda **kwargs:SimpleNamespace(logits=logits.bfloat16())
        with self.assertRaises(ValueError):r.likelihood([],'AB')
if __name__=='__main__':unittest.main()
