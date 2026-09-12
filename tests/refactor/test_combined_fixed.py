import math
import unittest
from types import SimpleNamespace
from llm_mas.runtime import Runtime
from llm_mas.config import resolve
from llm_mas.prompts import templates, for_condition
from llm_mas.engine import conversation

class Tokenizer:
    def __call__(self,text):return {'input_ids':[ord(c) for c in text]}
    def convert_ids_to_tokens(self,ids):return [chr(i) for i in ids]

class Fake(Runtime):
    def __init__(self):self.tokenizer=Tokenizer();self.calls=[]
    def likelihood(self,messages,candidate,continuation_prefix=''):
        self.calls.append((messages,candidate,continuation_prefix))
        # S2 candidate wins; second S2 argument wins, independent of list cycling.
        score=0 if candidate.startswith('I support the strategy with less') or candidate.startswith('This strategy reduces') else -5
        return score,[score],'prompt'+continuation_prefix

class CombinedTests(unittest.TestCase):
    def test_shared_prompt_and_exact_text(self):
        for schedule in ['synchronous','sequential','reverse_sequential','no_peer']:
            c=resolve(dict(argument_mode='combined_fixed',rounds=1,schedule=schedule))
            f=Fake();rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:f})
            for r in rows:
                self.assertEqual(r['strategy'],2)
                sel=r['argument_selection'];self.assertEqual(sel['prompt'],r['decision']['probe'])
                self.assertEqual(r['message'],sel['continuation_prefix']+sel['argument'])
                self.assertIn(sel['argument'],c['scenarios'][0]['strategy_2']['arguments'])
                for k in [1,2]:
                    for a in c['scenarios'][0][f'strategy_{k}']['arguments']:self.assertIn(a,sel['prompt'])
                self.assertNotIn(sel['prompt'],[m['content'] for m in r['history']])
            self.assertTrue(any(prefix for _,_,prefix in f.calls))
            self.assertTrue(all(text in c['scenarios'][0]['strategy_2']['arguments'] for _,text,prefix in f.calls if prefix))

    def test_forced_and_braces(self):
        c=resolve(dict(argument_mode='combined_fixed',rounds=1,initialization=[1,2]))
        c['scenarios'][0]['strategy_1']['arguments'][0]='Literal {example}.'
        f=Fake();rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:f})
        self.assertEqual(rows[0]['strategy'],1)
        self.assertIsNone(rows[0]['decision'])
        self.assertIn('Literal {example}.',rows[0]['argument_selection']['prompt'])
        self.assertEqual(templates()['stance_probe'],for_condition(templates(),{'argument_mode':'generated'},c['scenarios'][0])['stance_probe'])

    def test_continuation_excludes_prefix_tokens(self):
        import torch
        r=Runtime.__new__(Runtime);r.torch=torch;r.tokenizer=Tokenizer()
        # inputs normally returns a mapping supporting attribute access.
        class Inputs(dict):
            @property
            def input_ids(self):return self['input_ids']
        r.inputs=lambda messages,candidate:('P',Inputs(input_ids=torch.tensor([[ord(c) for c in 'P'+candidate]])))
        r.model=lambda **kw:SimpleNamespace(logits=torch.zeros(1,kw['input_ids'].shape[1],128))
        score,values,rendered=r.likelihood([], 'abc',continuation_prefix='candidate\n')
        self.assertEqual(len(values),3)
        self.assertAlmostEqual(score,-3*math.log(128),places=5)
        self.assertEqual(rendered,'Pcandidate\n')
