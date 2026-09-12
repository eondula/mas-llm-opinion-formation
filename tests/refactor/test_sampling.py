import unittest
from llm_mas.config import resolve
from llm_mas.engine import conversation
from llm_mas.prompts import templates

class Fake:
    def __init__(self):self.options=[]
    def measure(self,h,s,t):return dict(margin=1.,stance=.46)
    def generate(self,h,n,continuation_prefix='',**kwargs):
        self.options.append(kwargs)
        return dict(output='A reason.')

class SamplingTests(unittest.TestCase):
    def test_fixed_sampling_seeds_and_choices(self):
        c=resolve(dict(decoding='fixed_sampling',argument_mode='combined_reasoning',rounds=2))
        f=Fake();rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:f})
        self.assertEqual([o['sampling']['seed'] for o in f.options],list(range(42,48)))
        self.assertTrue(all(o['sampling']['temperature']==.7 and o['sampling']['top_p']==.9 for o in f.options))
        self.assertTrue(all(r['strategy']==1 for r in rows))
        f=Fake();c=resolve(dict(argument_mode='combined_reasoning',rounds=1))
        conversation(c,c['scenarios'][0],templates(),{c['models'][0]:f})
        self.assertTrue(all(not o for o in f.options))
    def test_editable_methods_and_validation(self):
        base=dict(decoding='fixed_sampling',argument_mode='combined_reasoning',sampling=dict(temperature=.4,top_p=.8,seed=123))
        for method,temp,top in [('temperature',.4,1.),('top_p',1.,.8),('combined',.4,.8)]:
            c=resolve(dict(base,sampling_method=method))
            self.assertEqual(c['sampling'],dict(temperature=temp,top_p=top,seed=123,top_k=0))
            self.assertEqual(resolve(c)['sampling'],c['sampling'])
        for invalid in [dict(temperature=0),dict(temperature=float('nan')),dict(top_p=1.2),dict(seed=-1),dict(seed=1.5)]:
            with self.assertRaises(ValueError):resolve(dict(base,sampling=invalid))

    def test_fixed_list_rejected(self):
        with self.assertRaises(ValueError):resolve(dict(decoding='fixed_sampling',argument_mode='predefined'))
