import importlib.util
from pathlib import Path
import unittest
import torch
spec=importlib.util.spec_from_file_location('precision',Path(__file__).with_name('run.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_same_logits_two_reductions(self):
        logits=torch.tensor([[[8.,7.9,0.],[1.,1.2,0.]]],dtype=torch.bfloat16);copy=logits.clone();ids=torch.tensor([[0,1]])
        old,new=m.reductions(torch,logits,ids)
        self.assertTrue(torch.equal(logits,copy))
        expected=torch.log_softmax(logits.float(),-1).gather(-1,ids.unsqueeze(-1)).squeeze(-1)[0]
        self.assertEqual(new['score'],expected.sum(dtype=torch.float32).item())
        self.assertEqual(old['score'],sum(torch.log_softmax(logits,-1).gather(-1,ids.unsqueeze(-1)).squeeze(-1)[0].tolist()))
        self.assertEqual(old['logsoftmax_dtype'],'torch.bfloat16');self.assertEqual(new['logsoftmax_dtype'],'torch.float32')
    def test_float32_forward_scores_keep_dtype(self):
        outputs=m.reductions(torch,torch.randn(1,3,8,dtype=torch.float32),torch.tensor([[1,2,3]]))
        self.assertTrue(all(x['logits_dtype']=='torch.float32' and x['gathered_dtype']=='torch.float32' for x in outputs))
if __name__=='__main__':unittest.main()
