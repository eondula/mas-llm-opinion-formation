"""Synthetic module tests only: no pretrained weights or experiment execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import torch
spec=importlib.util.spec_from_file_location('components',Path(__file__).with_name('run.py'))
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)

class Attention(torch.nn.Module):
    def forward(self,x):return (2*x, None)
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=Attention();self.mlp=torch.nn.Identity()
    def forward(self,x):
        x=x+self.self_attn(x)[0]
        return x+self.mlp(x)

class ComponentTests(unittest.TestCase):
    def test_modules_and_component_boundaries(self):
        block=Block();runtime=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=[block])))
        modules=pilot.component_modules(runtime,[0])
        x=torch.ones(1,3,2);vectors={}
        with pilot.observe([m for _,_,m in modules],2,vectors):baseline=block(x)
        for index,value in enumerate((2,3,6)):
            self.assertTrue(torch.equal(vectors[index],torch.full((2,),float(value))))
        for index,(_,component,module) in enumerate(modules):
            with pilot.patch(module,2,vectors[index]):self.assertTrue(torch.equal(block(x),baseline))
            self.assertFalse(module._forward_hooks)
        with pilot.patch(block.self_attn,2,torch.full((2,),4.)):
            output=block(x)
        self.assertTrue(torch.equal(output[0,2],torch.full((2,),10.)))
        self.assertTrue(torch.equal(output[0,:2],baseline[0,:2]))
        with pilot.patch(block.mlp,2,torch.full((2,),4.)):
            output=block(x)
        self.assertTrue(torch.equal(output[0,2],torch.full((2,),7.)))
        self.assertFalse(block.mlp._forward_hooks)

    def test_cleanup_and_one_call_only(self):
        module=Attention();x=torch.ones(1,3,2)
        with pilot.patch(module,2,torch.zeros(2)):
            y=module(x)
            self.assertIsNone(y[1])
            self.assertTrue(torch.equal(y[0][0,2],torch.zeros(2)))
            self.assertTrue(torch.equal(module(x)[0],2*x))
        with self.assertRaises(RuntimeError):
            with pilot.patch(module,2,torch.zeros(2)):raise RuntimeError('abort')
        self.assertFalse(module._forward_hooks)

    def test_gpu_selection(self):
        fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False),backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:True)))
        self.assertEqual(pilot.pilot_device(fake,'auto'),'mps')
        fake.cuda.is_available=lambda:True
        self.assertEqual(pilot.pilot_device(fake,'auto'),'cuda')
        fake.cuda.is_available=lambda:False;fake.backends.mps.is_available=lambda:False
        with self.assertRaises(RuntimeError):pilot.pilot_device(fake,'auto')

if __name__=='__main__':unittest.main()
