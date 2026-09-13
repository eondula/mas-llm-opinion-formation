import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from llm_mas.config import resolve
from llm_mas.engine import conversation
from llm_mas.prompts import templates
from llm_mas.choice_output import parse
from llm_mas.full_study_controls import branches

class Runtime:
    def __init__(self,*args):pass
    def measure(self,*args):return dict(margin=-1.,stance=-.46)
    def generate(self,*args,**kwargs):return dict(output='{"strategy":1,"explanation":"Reason."}')

class FullStudyTests(unittest.TestCase):
    def test_schema(self):
        self.assertEqual(parse('{"strategy":1,"explanation":"yes"}')[0],1)
        for s in ['A','{"strategy":true,"explanation":"yes"}','{"strategy":3,"explanation":"yes"}']:
            self.assertIsNone(parse(s)[0])
    def test_fenced_json_and_malformed_json(self):
        text='{"strategy":1,"explanation":"Reason."}'
        self.assertEqual(parse('```json\n'+text+'\n```'),(1,'Reason.','valid'))
        self.assertIsNone(parse('```json\n'+text+'"\n```')[0])
        self.assertIsNone(parse(text+' extra')[0])

    def test_generated_choice_not_overridden_by_margin(self):
        c=resolve(dict(rounds=1,argument_mode='combined_reasoning',decision_mode='generated'))
        rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:Runtime()})
        self.assertTrue(all(r['strategy']==1 and r['decision']['margin']==-1 for r in rows))
    def test_observation_does_not_stop(self):
        c=resolve(dict(mode='no_fixed_rounds',rounds=3,argument_mode='combined_reasoning',observe_stopping=True))
        with patch('llm_mas.message_similarity.MessageSimilarity') as cls:
            cls.return_value.compare.return_value=[1.,1.];cls.return_value.encoder='test';cls.return_value.revision='test'
            rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:Runtime()})
        self.assertEqual(len(rows),8)
        self.assertEqual(rows[2]['termination']['reason'],'system_detected_stability')
    def test_controls_preserve_input(self):
        row=dict(deliveries=[dict(speaker='Agent B',message='Endorse\nReason')],decision_history=[{'role':'system','content':'role'},{'role':'assistant','content':'self'},{'role':'user','content':'Agent B said:\nEndorse\nReason'}])
        output=list(branches(row));self.assertEqual(len(output),8)
        self.assertEqual(output[0][2],row['decision_history'])
        self.assertEqual(len(output[-1][2]),1)
        self.assertEqual(len(row['decision_history']),3)
    def test_pipeline_exports(self):
        from llm_mas.runner import execute
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'run'
            with patch('llm_mas.runner.run_path',return_value=path),patch('llm_mas.runtime.Runtime',Runtime),patch('llm_mas.message_similarity.MessageSimilarity') as cls:
                cls.return_value.compare.side_effect=lambda a,b:[1.]*len(a)
                cls.return_value.encoder='test';cls.return_value.revision='test'
                execute(dict(mode='study',rounds=2,argument_mode='combined_reasoning',decision_mode='generated',stopping='sign_stability',sentiment=False,export_profile='full_behavioral'),name='test')
            self.assertTrue((path/'figures/index.html').exists())
            self.assertTrue((path/'tables/outcomes.csv').exists())
            self.assertTrue((path/'paper_outputs.zip').exists())
            self.assertEqual(json.loads((path/'status.json').read_text())['phase'],'complete')

    def test_reversed_generated_choice_parses_fences_and_preserves_blocks(self):
        from llm_mas.web import display_event
        class FencedRuntime(Runtime):
            def generate(self, *args, **kwargs):
                return dict(output='```json\n{"strategy":2,"explanation":"Reason."}\n```')
        raw = json.loads(Path('studies/full_behavioral_reversed.json').read_text())
        raw.update(rounds=1, decision_mode='generated', stopping='fixed')
        c = resolve(raw)
        rows = conversation(c,c['scenarios'][0],templates(),{c['models'][0]:FencedRuntime()})
        for row in rows:
            instruction = row['generation']['instruction']
            self.assertLess(instruction.index('STRATEGY 2'), instruction.index('STRATEGY 1'))
            self.assertEqual(row['strategy'], 2)
            self.assertEqual(row['choice_status'], 'valid')
            self.assertEqual(display_event(dict(task='reversed', row=row))['row']['display_message'], 'Strategy 2\nReason.')
            self.assertEqual(row['message'], row['generation']['output'])

    def test_missing_final_brace_recovery_is_narrow_and_audited(self):
        from llm_mas.choice_output import recovery
        raw = '{"strategy":2,"explanation":"Reason."'
        self.assertEqual(parse(raw), (2, 'Reason.', 'valid'))
        self.assertEqual(recovery(raw), 'missing_final_object_brace')
        self.assertIsNone(recovery(raw + '}'))
        for invalid in ['{"strategy":2,"explanation":"unfinished',
                        '{"strategy":3,"explanation":"Reason."',
                        '{"strategy":true,"explanation":"Reason."',
                        '{"strategy":2,"explanation":"Reason.","extra":"x"',
                        'prose ' + raw]:
            self.assertIsNone(parse(invalid)[0])

    def test_recovered_generated_choice_preserves_raw_history(self):
        class MissingBraceRuntime(Runtime):
            def generate(self, *args, **kwargs):
                return dict(output='{"strategy":2,"explanation":"Reason."')
        c=resolve(dict(rounds=1,argument_mode='combined_reasoning',decision_mode='generated',reverse_candidate_blocks=True,presentation_order='strategy_2_first'))
        rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:MissingBraceRuntime()})
        for row in rows:
            self.assertEqual(row['strategy'],2)
            self.assertEqual(row['generation']['choice_recovery'],'missing_final_object_brace')
            self.assertEqual(row['history'][-1]['content'],row['generation']['output'])
