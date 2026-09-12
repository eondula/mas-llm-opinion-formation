import unittest
from llm_mas.config import resolve
from llm_mas.engine import conversation
from llm_mas.planning import configurations

from test_combined_fixed import Fake
from llm_mas.prompts import templates

class GeneratingFake(Fake):
    def __init__(self):super().__init__();self.generated=[]
    def generate(self,messages,max_tokens,continuation_prefix=''):
        self.generated.append((messages,max_tokens,continuation_prefix))
        return {'output':'First reason, combined with a second reason.'}

class CombinedReasoningTests(unittest.TestCase):
    def test_shared_prompt_generation_and_history(self):
        for schedule in ['synchronous','sequential','reverse_sequential','no_peer']:
            c=resolve(dict(argument_mode='combined_reasoning',rounds=1,schedule=schedule))
            f=GeneratingFake();rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:f})
            self.assertEqual(len(f.generated),len(rows))
            for r,(messages,limit,prefix) in zip(rows,f.generated):
                self.assertEqual(r['strategy'],2)
                self.assertEqual(messages[-1]['content'],r['decision']['probe'])
                self.assertEqual(messages[:-1],r['decision_history'])
                self.assertEqual(r['message'],prefix+r['argument'])
                self.assertIsNone(r['argument_index']);self.assertIsNone(r['argument_selection'])
                self.assertEqual(r['history'][-1]['content'],r['message'])
                self.assertEqual(limit,c['max_argument_tokens'])
                for k in (1,2):
                    for a in c['scenarios'][0][f'strategy_{k}']['arguments']:self.assertIn(a,messages[-1]['content'])
            # Only candidate scores, never individual argument scores.
            self.assertTrue(all(candidate.startswith('I support') and not prefix for _,candidate,prefix in f.calls))
    def test_forced_and_study_condition(self):
        c=resolve(dict(mode='study',rounds=1,initialization=[1,2],conditions=[{'argument_mode':'combined_reasoning'}]))
        c=resolve(next(configurations(c)));f=GeneratingFake()
        rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:f})
        self.assertEqual(rows[0]['strategy'],1);self.assertIsNone(rows[0]['decision'])
        self.assertTrue(f.generated[0][2].startswith(c['scenarios'][0]['strategy_1']['candidate']))
