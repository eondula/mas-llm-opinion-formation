import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from llm_mas.config import resolve
from llm_mas.engine import conversation
from llm_mas.prompts import templates
from llm_mas.runner import write
from llm_mas.saved_analysis import execute, validate, catalog
from tests.refactor.test_pipeline import FakeRuntime


class SavedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.base=self.root/'results/studies'
        self.patch=patch('llm_mas.saved_analysis.run_path',side_effect=lambda name:self.base/name)
        self.patch.start();self.addCleanup(self.patch.stop)
        self.rootpatch=patch('llm_mas.saved_analysis.ROOT',self.root)
        self.rootpatch.start();self.addCleanup(self.rootpatch.stop)
        self.config=resolve(dict(rounds=1,sentiment=False))
        self.texts=templates()
        self.rows=conversation(self.config,self.config['scenarios'][0],self.texts,{self.config['models'][0]:FakeRuntime()})
        for name in ('one','two'):
            write(self.base/name/'config.json',self.config)
            write(self.base/name/'inputs/prompts.json',self.texts)
            write(self.base/name/'conversations/000_000.json',dict(config=self.config,scenario=self.config['scenarios'][0],rows=self.rows))
        self.options=dict(source_run='one',task='000_000',compare_run='two',compare_task='000_000',modules=[])

    def test_exports_and_comparison_leave_source_unchanged(self):
        original={str(p):p.read_bytes() for p in (self.base/'one').rglob('*') if p.is_file()}
        with patch('llm_mas.runtime.Runtime',side_effect=AssertionError('No models for exports')):
            path=execute(self.options,'derived')
        self.assertEqual(original,{str(p):p.read_bytes() for p in (self.base/'one').rglob('*') if p.is_file()})
        with (path/'tables/comparison_matched_rounds.csv').open() as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),4)
        self.assertTrue(all(float(r['post_margin_B_minus_A'])==0 for r in rows))
        from llm_mas.saved_analysis import list_analyses
        self.assertEqual([r['run_id'] for r in list_analyses()],['derived'])
        info=catalog('derived')
        self.assertTrue(info['figures'])
        self.assertTrue(all(f['csv'] for f in info['figures'] if not f['name'].endswith('framework.png')))
        self.assertTrue(info['tables'])
        self.assertEqual(info['status']['phase'],'complete')
        with (path/'figures/000_000_stance_sentiment.csv').open() as f:
            points=list(csv.DictReader(f))
        self.assertEqual(float(points[0]['y']),self.rows[0]['post']['stance'])

    def test_selected_replay_only(self):
        opts=dict(self.options,modules=['replay'],agent='Agent B',round=1)
        with patch('llm_mas.runtime.Runtime',FakeRuntime):path=execute(opts,'replayed')
        rows=json.loads((path/'measurements/replay.json').read_text())
        self.assertEqual([(r['agent'],r['round']) for r in rows],[('Agent B',1)])

    def test_reject_invalid_selection_and_scenario(self):
        with self.assertRaises(ValueError):validate(dict(self.options,task='../x'))
        with self.assertRaises(ValueError):validate(dict(self.options,modules=['heads'],agent='Agent A',round=0))
        record=json.loads((self.base/'two/conversations/000_000.json').read_text())
        record['scenario']['strategy_1']['candidate']='Different meaning'
        write(self.base/'two/conversations/000_000.json',record)
        with self.assertRaisesRegex(ValueError,'same scenario'):validate(self.options)

    def test_default_all_layers_and_continuation_sweep(self):
        from types import SimpleNamespace
        from llm_mas.research import run
        runtime=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=[0,1,2])))
        config=dict(self.config,research=['mechanistic','heads','continuation'],
                    research_selection={'agent':'Agent B','round':1},research_events_per_conversation=1,
                    patch_layers=None,continuation_all_layers=True)
        with patch('llm_mas.interventions.sweep',return_value=[]) as components, patch('llm_mas.interventions.head_sweep',return_value=[]) as heads, patch('llm_mas.continuation.evaluate',return_value={}) as downstream:
            run(self.base/'one',config,self.texts,{self.config['models'][0]:runtime},lambda *args:None)
        self.assertIsNone(components.call_args.args[-1])
        self.assertIsNone(heads.call_args.args[-1])
        self.assertEqual([call.args[-1] for call in downstream.call_args_list],[0,1,2])
