import importlib.util
from pathlib import Path
import unittest
import torch
spec=importlib.util.spec_from_file_location('pilot',Path(__file__).with_name('run.py'))
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)

class DeviceTests(unittest.TestCase):
    def test_cuda_default_never_falls_back(self):
        from types import SimpleNamespace
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False), backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:False)))
        for requested in ('cuda','mps','auto'):
            with self.assertRaisesRegex(RuntimeError,'automatic CPU fallback is disabled'):
                pilot.pilot_device(fake,requested)
        self.assertEqual(pilot.pilot_device(fake,'cpu'),'cpu')
        fake.backends.mps.is_available=lambda:True
        self.assertEqual(pilot.pilot_device(fake,'auto'),'mps')
        self.assertEqual(pilot.pilot_device(fake,'mps'),'mps')
        with self.assertRaises(RuntimeError):pilot.pilot_device(fake,'cuda')
        fake.cuda.is_available=lambda:True
        self.assertEqual(pilot.pilot_device(fake,'auto'),'cuda')
        self.assertEqual(pilot.pilot_device(fake,'cuda'),'cuda')

class HookTests(unittest.TestCase):
    def test_patch_only_first_call_and_cleanup(self):
        block=torch.nn.Identity();x=torch.zeros(1,4,3);donor=torch.ones(3)
        with pilot.patch(block,2,donor):
            y=block(x)
            self.assertTrue(torch.equal(y[0,2],donor))
            self.assertTrue(torch.equal(y[0,1],x[0,1]))
            self.assertTrue(torch.equal(block(torch.zeros(1,1,3)),torch.zeros(1,1,3)))
        self.assertFalse(block._forward_hooks)
        self.assertTrue(torch.equal(block(x),x))

    def test_cleanup_on_error(self):
        block=torch.nn.Identity()
        with self.assertRaises(RuntimeError):
            with pilot.patch(block,0,torch.ones(3)):raise RuntimeError('test')
        self.assertFalse(block._forward_hooks)

    def test_observation_and_self_replacement_are_inert(self):
        block=torch.nn.Identity();x=torch.randn(1,4,3);vectors={}
        with pilot.observe([block],3,vectors):self.assertTrue(torch.equal(block(x),x))
        self.assertFalse(block._forward_hooks)
        with pilot.patch(block,3,vectors[0]):self.assertTrue(torch.equal(block(x),x))

if __name__=='__main__':unittest.main()
