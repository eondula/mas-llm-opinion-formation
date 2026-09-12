import unittest
from llm_mas.config import resolve
from llm_mas.prompts import initial, templates
from llm_mas.engine import conversation

class Fake:
    def measure(self,h,s,t):return {'margin':1,'stance':.46}
    def generate(self,h,t,continuation_prefix=''):return {'output':'A reason.'}

class PersonaTests(unittest.TestCase):
    def test_preset_and_persistent_roles(self):
        c=resolve(dict(preview_preset='persona',scenarios=['sensor'],rounds=1,argument_mode='combined_reasoning',initialization=[1,2]))
        self.assertEqual((c['rounds'],c['agents'],c['argument_mode'],c['max_argument_tokens']),(1,2,'combined_reasoning',128))
        self.assertTrue(c['sentiment'])
        s=c['scenarios'][0];t=templates()
        plain=initial(s,t)
        a=initial(s,t,persona=1);b=initial(s,t,persona=2)
        self.assertEqual(a[1],plain[1]);self.assertNotEqual(a[0],b[0])
        rows=conversation(dict(c,rounds=1),s,t,{c['models'][0]:Fake()},lambda **kwargs:None)
        self.assertEqual([r['strategy'] for r in rows[:2]],[1,2])
        for row in rows:
            role=s['persona_roles']['strategy_1' if row['agent']=='Agent A' else 'strategy_2']['role_name']
            self.assertIn(role,row['history'][0]['content'])
    def test_invalid_schedule_and_roles(self):
        c=resolve(dict(preview_preset='persona',agents=4,schedule='no_peer',models=['custom']))
        self.assertEqual(c['agents'],4)
        self.assertEqual(c['schedule'],'no_peer')
        self.assertEqual(c['models'],['custom'])
        self.assertFalse(resolve(dict(c, preview_preset='no_persona'))['persona'])
        s=resolve({})['scenarios'][0];s.pop('persona_roles',None)
        with self.assertRaises(ValueError):resolve(dict(preview_preset='persona',scenarios=[s]))
