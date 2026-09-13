import unittest
from llm_mas.web import display_event

class PayloadTests(unittest.TestCase):
    def test_playback_fields_without_full_history(self):
        row=dict(round=1,agent='Agent A',model='toy',strategy=1,forced=False,
                 message='A reason',deliveries=[{'speaker':'Agent B','message':'B reason'}],
                 sentiment=.2,history=['large'],decision={'rendered_prompt':'large'},
                 post={'stance':.5,'rendered_prompt':'large'})
        compact=display_event(dict(task='000_000',row=row))
        self.assertEqual(compact['row']['post'],{'stance':.5})
        self.assertEqual(compact['row']['deliveries'],row['deliveries'])
        self.assertNotIn('history',compact['row'])
        self.assertNotIn('decision',compact['row'])
        self.assertIn('history',row)

    def test_legacy_stopping_badges_are_not_exposed(self):
        row=dict(round=1,agent='Agent B',post={},termination={'decisions':[
            {'agent':'Agent B','action':'CONTINUE','valid':False}]})
        compact=display_event(dict(task='000',row=row))
        self.assertNotIn('stopping',compact['row'])

    def test_generated_display_handles_legacy_fences_without_repairing_record(self):
        raw = '```json\n{"strategy":2,"explanation":"Coverage matters."}\n```'
        row = dict(message=raw, strategy=None, argument=raw, choice_status='invalid_json')
        shown = display_event(dict(task='reversed', row=row))['row']
        self.assertTrue(shown['display_message'].startswith('Strategy 2\nCoverage matters.'))
        self.assertIn('record unchanged', shown['display_message'])
        self.assertIsNone(shown['strategy'])
        self.assertEqual(shown['message'], raw)
        self.assertEqual(row['choice_status'], 'invalid_json')

    def test_generated_display_keeps_malformed_output_visible(self):
        raw = '{"strategy":2,"explanation":"truncated'
        shown = display_event(dict(task='reversed', row=dict(message=raw, strategy=None)))['row']
        self.assertEqual(shown['display_message'], raw)

    def test_generated_display_uses_semantic_strategy_number(self):
        row = dict(message='{"strategy":2,"explanation":"Reason."}', strategy=2, choice_status='valid')
        shown = display_event(dict(task='reversed', row=row))['row']
        self.assertEqual(shown['display_message'], 'Strategy 2\nReason.')
