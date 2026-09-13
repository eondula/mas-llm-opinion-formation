import importlib.util
from pathlib import Path
import unittest
import torch
spec=importlib.util.spec_from_file_location('heads',Path(__file__).with_name('run.py'))
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)

class Tests(unittest.TestCase):
    def test_single_head_keeps_other_inputs_unchanged(self):
        projection=torch.nn.Identity();x=torch.arange(24.).reshape(1,3,8);donor=torch.full((8,),-1.)
        with pilot.replace_heads(projection,2,donor,[1],2):
            actual=projection(x)
            expected=x.clone();expected[:,2,2:4]=-1
            self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.equal(projection(x),x))
        self.assertFalse(projection._forward_pre_hooks)
        self.assertTrue(torch.equal(x,torch.arange(24.).reshape(1,3,8)))

    def test_all_heads_matches_projected_output(self):
        torch.manual_seed(1);projection=torch.nn.Linear(8,5,bias=True)
        x=torch.randn(1,3,8);donor=torch.randn(8)
        with pilot.replace_heads(projection,2,donor,list(range(4)),2):result=projection(x)
        self.assertTrue(torch.allclose(result[0,2],projection(donor),atol=1e-6))
        self.assertTrue(torch.equal(result[0,:2],projection(x)[0,:2]))

    def test_cleanup_on_exception(self):
        projection=torch.nn.Identity()
        with self.assertRaises(RuntimeError):
            with pilot.replace_heads(projection,1,torch.ones(4),[0],2):raise RuntimeError('abort')
        self.assertFalse(projection._forward_pre_hooks)
        with self.assertRaises(ValueError):
            with pilot.replace_heads(projection,1,torch.ones(4),[0],2):pass

    def test_combination_requires_two_and_uses_fixed_context(self):
        rows=[dict(case='changed',direction='reversed_to_forward',kind='single',control='donor',margin_delta=d,heads=str(h)) for h,d in enumerate([.01,.04,.03,-.1])]
        self.assertEqual(pilot.choose_combination(rows,.02),[1,2])
        self.assertEqual(pilot.choose_combination(rows,.035),[])
        rows[1]['case']='unchanged'
        self.assertEqual(pilot.choose_combination(rows,.02),[])

if __name__=='__main__':unittest.main()
