import unittest
from llm_mas.config import resolve
from llm_mas.engine import conversation
from llm_mas.prompts import templates

class Agent:
    def __init__(self, messages, margin=1): self.messages=iter(messages); self.margin=margin
    def measure(self,*args): return {'margin':self.margin,'stance':.4}
    def generate(self,*args,**kwargs): return {'output':next(self.messages)}

class StoppingTests(unittest.TestCase):
    def run_case(self, messages, cap=3, schedule='synchronous', fixed=False):
        n=len(messages)
        c=resolve(dict(mode='preview' if fixed else 'no_fixed_rounds',models=[str(i) for i in range(n)],agents=n,rounds=cap,schedule=schedule,argument_mode='generated'))
        if not fixed: c['stopping']='exact_repetition'
        agents={str(i):Agent(m,1 if i%2==0 else -1) for i,m in enumerate(messages)}
        return conversation(c,c['scenarios'][0],templates(),agents)
    def test_repeated_round_all_schedules_and_agent_counts(self):
        for schedule in ['synchronous','sequential','reverse_sequential','no_peer']:
            for n in [2,3,4]:
                rows=self.run_case([['same','same'] for _ in range(n)],schedule=schedule)
                self.assertEqual(len(rows),2*n)
                self.assertEqual(rows[-1]['termination']['reason'],'system_detected_repetition')
                self.assertNotIn('decisions',rows[-1]['termination'])
    def test_one_agent_change_prevents_stop(self):
        rows=self.run_case([['same']*3,['one','two','two']])
        self.assertEqual(len(rows),6)
        self.assertEqual(rows[-1]['termination']['reason'],'system_detected_repetition')
    def test_maximum_and_nonadjacent_repeat(self):
        rows=self.run_case([['a','b','a'],['a','b','a']],cap=2)
        self.assertEqual(rows[-1]['termination']['reason'],'max_rounds')
    def test_whitespace_is_not_normalized(self):
        rows=self.run_case([['a','a '],['b','b']],cap=1)
        self.assertEqual(rows[-1]['termination']['reason'],'max_rounds')
    def test_fixed_rounds_unchanged(self):
        rows=self.run_case([['a']*4,['a']*4],fixed=True)
        self.assertEqual(len(rows),8)
        self.assertNotIn('termination',rows[-1])
    def test_invalid_cap(self):
        with self.assertRaises(ValueError): resolve(dict(mode='no_fixed_rounds',rounds=0))

    def test_cosine_all_agents_threshold_inclusive(self):
        from unittest.mock import patch
        c=resolve(dict(mode='preview',stopping='cosine_similarity',models=['a','b'],rounds=3,argument_mode='generated'))
        with patch('llm_mas.message_similarity.MessageSimilarity') as cls:
            cls.return_value.compare.side_effect=[[.8,.95]]
            cls.return_value.encoder='test'
            cls.return_value.revision='test'
            rows=conversation(c,c['scenarios'][0],templates(),{'a':Agent(['one','two','three']),'b':Agent(['four','five','six'])})
        self.assertEqual(len(rows),6)
        self.assertIsNone(rows[2]['termination']['reason'])
        self.assertEqual(rows[-1]['termination']['reason'],'system_detected_repetition')
        self.assertEqual(rows[-1]['termination']['threshold'],.8)
        self.assertEqual(rows[-1]['termination']['similarities']['Agent A'],.8)

    def test_invalid_similarity_threshold(self):
        for threshold in [True, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                resolve(dict(mode='no_fixed_rounds',similarity_threshold=threshold))

    def test_minimum_defers_similarity_check(self):
        from unittest.mock import patch
        c=resolve(dict(mode='preview',stopping='cosine_similarity',rounds=1,argument_mode='generated'))
        with patch('llm_mas.message_similarity.MessageSimilarity') as cls:
            rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:Agent(['same']*4)})
            cls.assert_not_called()
        self.assertEqual(rows[-1]['termination']['reason'],'max_rounds')
        self.assertEqual(c['minimum_rounds'],2)

    def test_invalid_minimum(self):
        for value in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                resolve(dict(mode='preview',stopping='cosine_similarity',minimum_rounds=value))

    def test_combined_checks_start_round_one(self):
        from unittest.mock import patch
        c=resolve(dict(mode='no_fixed_rounds',rounds=3,argument_mode='generated',minimum_rounds=9))
        self.assertNotIn('minimum_rounds',c)
        with patch('llm_mas.message_similarity.MessageSimilarity') as cls:
            cls.return_value.compare.return_value=[.9,.9]
            cls.return_value.encoder='test';cls.return_value.revision='test'
            rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:Agent(['same']*4)})
        self.assertEqual(len(rows),4)
        self.assertEqual(rows[-1]['termination']['reason'],'system_detected_stability')

    def test_same_sign_margin_motion_allows_stop(self):
        from unittest.mock import patch
        class Moving(Agent):
            def measure(self,*args):
                self.margin+=1
                return {'margin':self.margin,'stance':.9}
        c=resolve(dict(mode='no_fixed_rounds',rounds=2,argument_mode='generated'))
        with patch('llm_mas.message_similarity.MessageSimilarity') as cls:
            cls.return_value.compare.return_value=[1.,1.]
            cls.return_value.encoder='test';cls.return_value.revision='test'
            rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:Moving(['same']*6)})
        self.assertEqual(rows[-1]['termination']['reason'],'system_detected_stability')

    def test_bad_margin_tolerance(self):
        for value in [-1,True,float('nan')]:
            with self.assertRaises(ValueError):resolve(dict(mode='preview',stopping='combined_stability',margin_tolerance=value))

    def test_strategy_switch_blocks_stop_even_when_post_margin_stable(self):
        from unittest.mock import patch
        class Switching(Agent):
            def __init__(self): super().__init__(['same']*4); self.calls=0
            def measure(self,*args):
                self.calls+=1
                return {'margin':-1 if self.calls==4 else 1,'stance':.4}
        c=resolve(dict(mode='no_fixed_rounds',rounds=1,models=['a','b'],argument_mode='generated'))
        with patch('llm_mas.message_similarity.MessageSimilarity') as cls:
            cls.return_value.compare.return_value=[1.,1.]
            cls.return_value.encoder='test';cls.return_value.revision='test'
            rows=conversation(c,c['scenarios'][0],templates(),{'a':Switching(),'b':Switching()})
        self.assertEqual(rows[-1]['termination']['reason'],'max_rounds')
        self.assertFalse(rows[-1]['termination']['strategy_unchanged']['Agent A'])
        self.assertEqual(rows[-1]['termination']['margin_changes']['Agent A'],0)

    def test_post_sign_change_blocks_stop(self):
        from unittest.mock import patch
        class SignFlip(Agent):
            def __init__(self): super().__init__(['same']*4); self.calls=0
            def measure(self,*args):
                self.calls+=1
                return {'margin':-.001 if self.calls==5 else 0.,'stance':0.}
        c=resolve(dict(mode='no_fixed_rounds',rounds=1,models=['a','b'],argument_mode='generated'))
        with patch('llm_mas.message_similarity.MessageSimilarity') as cls:
            cls.return_value.compare.return_value=[1.,1.]
            cls.return_value.encoder='test';cls.return_value.revision='test'
            rows=conversation(c,c['scenarios'][0],templates(),{'a':SignFlip(),'b':SignFlip()})
        self.assertEqual(rows[-1]['termination']['reason'],'max_rounds')
        self.assertFalse(rows[-1]['termination']['margin_sign_unchanged']['Agent A'])
        self.assertTrue(rows[-1]['termination']['strategy_unchanged']['Agent A'])
