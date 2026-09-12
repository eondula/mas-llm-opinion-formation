import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from llm_mas.config import resolve, ROOT
from llm_mas.planning import configurations

class Fake:
    def __init__(self,*args):pass
    def measure(self,h,s,t):return dict(margin=1,stance=.46)
    def generate(self,h,t,continuation_prefix=''):return dict(output='A reason and another reason.')

class ExperimentCTests(unittest.TestCase):
    def test_preset_scope(self):
        config=resolve(json.loads((ROOT/'studies/experiment_C.json').read_text()))
        tasks=list(configurations(config))
        self.assertEqual(len(tasks),16)
        self.assertEqual(sum(len(t['scenarios']) for t in tasks),32)
        self.assertEqual({t['schedule'] for t in tasks},{'synchronous','sequential'})
        self.assertTrue(all(t['argument_mode']=='combined_reasoning' and t['initialization'] in ('natural',[1,2]) for t in tasks))
        mixed={tuple(t['models']) for t in tasks if len(set(t['models']))>1}
        self.assertEqual(mixed,{('Qwen/Qwen2.5-1.5B-Instruct','HuggingFaceTB/SmolLM2-1.7B-Instruct')})
        ui=resolve(dict(mode='experiment_c',agents=4,rounds=3,scenarios=['storage'],argument_mode='generated',schedule='reverse_sequential',initialization=[2,1,2,1]))
        self.assertEqual(ui['agents'],4);self.assertEqual(ui['rounds'],3)
        self.assertEqual(len(list(configurations(ui))),1)
        self.assertEqual(ui['argument_mode'],'generated')
        self.assertEqual(ui['schedule'],'reverse_sequential')
        self.assertEqual(ui['initialization'],[2,1,2,1])
        self.assertNotIn('export_profile',ui)

    def test_saved_outputs_png_only(self):
        from llm_mas.runner import execute
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run'
            with patch('llm_mas.runner.run_path',return_value=path),patch('llm_mas.runtime.Runtime',Fake),patch('llm_mas.sentiment.measure_sentiment',return_value={'sentiment_score':.25}):
                execute(dict(json.loads((ROOT/'studies/experiment_C.json').read_text()),rounds=1,scenarios=['storage']),name='test_c')
            self.assertEqual(len(list((path/'conversations').glob('*.json'))),16)
            pngs=list((path/'figures').glob('*.png'));self.assertEqual(len(pngs),8)
            for png in pngs:
                with Image.open(png) as image:image.verify()
            self.assertFalse(list(path.rglob('*.pdf')))
            for kind,n in [('homogeneous',12),('heterogeneous',4)]:
                with (path/'tables'/f'{kind}_summary.csv').open() as f:rows=list(csv.DictReader(f))
                self.assertEqual(len(rows),n)
                self.assertTrue(all(r['final_outcome']=='agreement' and r['stable_agreement_round']==('0' if r['initialization']=='"natural"' else '1') for r in rows))
                self.assertTrue(all(r['agent_a_final_sentiment']=='0.25' for r in rows))
            with zipfile.ZipFile(path/'paper_outputs.zip') as z:
                self.assertIsNone(z.testzip());self.assertFalse(any(n.endswith('.pdf') for n in z.namelist()))
            self.assertEqual(json.loads((path/'status.json').read_text())['phase'],'complete')
