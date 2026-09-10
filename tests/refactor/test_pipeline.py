import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from llm_mas.validation import annotate_choice

class FakeRuntime:
    def __init__(self,*args):pass
    def measure(self,h,s,t):
        margin=.2+len(h)*.01
        return dict(margin=margin,stance=margin,likelihoods=[-.2,-.2-margin],token_log_probs=[[-.2],[-.2-margin]],rendered_prompt='fake')
    def generate(self,h,t):return dict(output='A. This option is preferred.')

class PipelineTests(unittest.TestCase):
    def test_independent_parser(self):
        self.assertEqual(annotate_choice('Option B: explanation'),'B')
        self.assertIsNone(annotate_choice('Both options have benefits'))
    def test_saved_pipeline(self):
        from llm_mas.runner import execute
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'run'
            with patch('llm_mas.runner.run_path',return_value=path),patch('llm_mas.runtime.Runtime',FakeRuntime),patch('llm_mas.sentiment.measure_sentiment',return_value={'sentiment_score':.25}):
                execute(dict(mode='study',rounds=1,sentiment=True,conditions=[{'initialization':'natural'},{'initialization':[2,1]}],research=['validation','history','replay','sufficiency','prediction']),name='test')
            for record in (path/'conversations').glob('*.json'):
                self.assertTrue(all(row['sentiment']==.25 for row in json.loads(record.read_text())['rows']))
            self.assertTrue((path/'paper_outputs.zip').exists())
            self.assertTrue((path/'tables/outcomes.csv').exists())
            self.assertTrue((path/'analysis/group_prediction.json').exists())
            self.assertEqual(json.loads((path/'status.json').read_text())['phase'],'complete')
            coverage=json.loads((path/'figures/coverage.json').read_text())
            self.assertTrue(coverage['complete_requested_suite'])
            self.assertFalse(coverage['complete_paper_suite'])
