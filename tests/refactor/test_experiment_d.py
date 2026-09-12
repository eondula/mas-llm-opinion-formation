import json
import tempfile
import unittest
from pathlib import Path
from llm_mas.config import ROOT,resolve
from llm_mas.planning import configurations
from llm_mas.engine import conversation
from llm_mas.prompts import templates

class Fake:
    def measure(self,h,s,t):return dict(margin=1.,stance=.462)
    def generate(self,h,n,continuation_prefix=''):return dict(output='This strategy preserves detail. It requires more resources.')

class ExperimentDTests(unittest.TestCase):
    def test_design(self):
        c=resolve(json.loads((ROOT/'studies/experiment_D.json').read_text()))
        tasks=[resolve(t) for t in configurations(c)]
        self.assertEqual(len(tasks),32)
        self.assertEqual(sum(len(t['scenarios']) for t in tasks),64)
        self.assertEqual(sum(len(t['scenarios'])*22 for t in tasks),1408)
        self.assertEqual({t['persona'] for t in tasks},{True,False})
        self.assertTrue(all(t['rounds']==10 and not t['research'] for t in tasks))

    def test_exports(self):
        from llm_mas.figures import render
        from PIL import Image
        c=resolve(json.loads((ROOT/'studies/experiment_D.json').read_text()))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'conversations').mkdir();(p/'config.json').write_text(json.dumps(c))
            for i,t in enumerate(configurations(c)):
                t=resolve(t)
                for j,s in enumerate(t['scenarios']):
                    rows=conversation(t,s,templates(),{m:Fake() for m in t['models']})
                    for r in rows:r['sentiment']=.2
                    (p/'conversations'/f'{i:03}_{j:03}.json').write_text(json.dumps(dict(config=t,scenario=s,rows=rows)))
            render(p)
            images=list((p/'figures').glob('*.png'))
            self.assertEqual(len(images),36)
            self.assertFalse(list(p.rglob('*.pdf')))
            for f in images:
                with Image.open(f) as im:im.verify()
            data=json.loads((p/'figures/coverage.json').read_text())
            self.assertEqual(data['responses'],1408)
            self.assertTrue((p/'tables/matched_comparisons.csv').exists())
