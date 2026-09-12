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
