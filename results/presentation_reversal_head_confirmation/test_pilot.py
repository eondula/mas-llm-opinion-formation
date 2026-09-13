import importlib.util
from pathlib import Path
import unittest
import torch
spec=importlib.util.spec_from_file_location('confirmation',Path(__file__).with_name('run.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_endpoints_and_midpoint_preserve_other_heads(self):
        module=torch.nn.Identity();x=torch.arange(12.).reshape(1,2,6);donor=torch.full((6,),20.)
        for alpha in (0,.5,1):
            with m.interpolate(module,1,donor,[0,2],[alpha,0],2):y=module(x)
            expected=x.clone();expected[:,1,:2]=(1-alpha)*x[:,1,:2]+alpha*20
            self.assertTrue(torch.equal(y,expected));self.assertFalse(module._forward_pre_hooks)
    def test_joint_and_cleanup(self):
        module=torch.nn.Identity();x=torch.zeros(1,2,6)
        with m.interpolate(module,1,torch.ones(6),[0,2],[1,.5],2):
            y=module(x);self.assertTrue(torch.equal(y[0,1],torch.tensor([1.,1.,0.,0.,.5,.5])))
            self.assertTrue(torch.equal(module(x),x))
        with self.assertRaises(RuntimeError):
            with m.interpolate(module,1,torch.ones(6),[0,2],[1,1],2):raise RuntimeError('abort')
        self.assertFalse(module._forward_pre_hooks)
if __name__=='__main__':unittest.main()
