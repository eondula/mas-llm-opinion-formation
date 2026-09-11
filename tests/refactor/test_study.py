import json
import tempfile
import unittest
from pathlib import Path
from llm_mas.config import resolve_run_id,resolve
from llm_mas.planning import configurations

class StudyTests(unittest.TestCase):
    def test_ids(self):
        self.assertEqual(resolve_run_id(' my_run '),'my_run')
        self.assertRegex(resolve_run_id(''),r'^\d{8}_\d{6}_\d{6}$')
        with self.assertRaises(ValueError):resolve_run_id('../escape')
    def test_expansion(self):
        c=resolve({'mode':'study','grid':{'initialization':['natural','all_forced'],'schedule':['synchronous']}})
        items=list(configurations(c))
        self.assertEqual(len(items),5)
        self.assertEqual(items[-1]['initialization'],[2,2])
    def test_paper_checkpoint_coverage(self):
        from itertools import product, permutations
        from llm_mas.config import ROOT
        raw=json.loads((ROOT/'studies/paper.json').read_text())
        items=list(configurations(resolve(raw)))
        models={'Qwen/Qwen2.5-1.5B-Instruct','Qwen/Qwen2.5-3B-Instruct','HuggingFaceTB/SmolLM2-1.7B-Instruct'}
        pairs={tuple(c['models']) for c in items if c['agents']==2}
        triples={tuple(c['models']) for c in items if c['agents']==3}
        self.assertEqual(pairs,set(product(models,repeat=2)))
        self.assertEqual(triples,set(permutations(models)))
        self.assertTrue(all(len(set(t))==3 for t in triples))
        self.assertEqual(sum(len(c['scenarios']) for c in items if c['agents']==2),1440)
        self.assertEqual(sum(len(c['scenarios']) for c in items if c['agents']==3),1728)
    def test_patch_cleanup(self):
        import torch
        from llm_mas.interventions import replacement
        m=torch.nn.Identity();x=torch.ones(1,2,3)
        with replacement(m,1,torch.zeros(3)):
            self.assertEqual(m(x)[0,1].sum().item(),0)
        self.assertEqual(m(x).sum().item(),6)
        self.assertEqual(len(m._forward_hooks),0)
    def test_prediction_no_task_leak(self):
        from llm_mas.prediction import run
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'conversations').mkdir()
            for i in range(3):
                rows=[dict(round=j,agent='Agent A',model='fake',before={'margin':j*.1},decision={'margin':j*.2},decision_history=[],deliveries=[]) for j in range(1,4)]
                (p/'conversations'/f'{i}.json').write_text(json.dumps({'rows':rows}))
            run(p)
            data=json.loads((p/'analysis/prediction.json').read_text())
            self.assertEqual(len(data['rows']),9)
            self.assertTrue(all(r['task'] not in r['train_tasks'] for r in data['rows']))
