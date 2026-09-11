import copy
import tempfile
import unittest
from pathlib import Path
from llm_mas.config import resolve,scenario
from llm_mas.prompts import templates,initial
from llm_mas.engine import conversation

class Fake:
    def measure(self,history,scenario,texts):
        return {'margin':.4,'stance':.2,'rendered_prompt':'test'}
    def generate(self,messages,max_tokens):return {'output':'Generated support.'}

class EngineTests(unittest.TestCase):
    def test_prompt_exact(self):
        s=scenario('storage')
        expected = (
            "Storage volume versus data fidelity.\n\n"
            "A research team must choose between two strategies.\n\n"
            "Option A:\n- more storage space is required\n- stored data has higher fidelity\n\n"
            "Option B:\n- less storage space is required\n- stored data has lower fidelity\n\n"
            "No other information should be considered."
        )
        self.assertEqual(initial(s,templates())[1]['content'],expected)
    def test_synchronous_uses_old_messages(self):
        c=resolve({'rounds':2,'initialization':[2,1]})
        rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:Fake()})
        a0,b0,a1,b1=rows[:4]
        self.assertIn(b0['message'],a1['decision_history'][-1]['content'])
        self.assertIn(a0['message'],b1['decision_history'][-1]['content'])
        self.assertEqual(a0['deliveries'], [])
        self.assertEqual(b1['deliveries'][0]['speaker'], 'Agent A')
        self.assertEqual(b1['deliveries'][0]['message'], a0['message'])
        self.assertIsNone(a0['decision'])
        self.assertNotIn('State your current position.',str(a1['history']))
    def test_sequential_uses_new_message(self):
        c=resolve({'rounds':1,'initialization':[2,1],'schedule':'sequential'})
        rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:Fake()})
        self.assertIn(rows[2]['message'],rows[3]['decision_history'][-1]['content'])
    def test_n_agents_models(self):
        c=resolve({'mode':'study','rounds':1,'agents':3,'models':['x','y','x']})
        rows=conversation(c,c['scenarios'][0],templates(),{'x':Fake(),'y':Fake()})
        self.assertEqual(len(rows),6)
        self.assertEqual(rows[4]['model'],'y')
        self.assertEqual(len(rows[3]['decision_history']),5)
    def test_generated(self):
        c=resolve({'rounds':1,'argument_mode':'generated'})
        rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:Fake()})
        self.assertEqual(rows[0]['argument'],'Generated support.')
    def test_invalid_preview(self):
        with self.assertRaises(ValueError):resolve({'scenarios':['sensor','storage']})
    def test_figures(self):
        import json
        from llm_mas.figures import render
        c=resolve({'rounds':1})
        rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:Fake()})
        for r in rows:r['sentiment']=.5
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'conversations').mkdir()
            (p/'conversations/000.json').write_text(json.dumps(dict(config=c,scenario=c['scenarios'][0],rows=rows)))
            render(p)
            self.assertTrue((p/'figures/000_stance_sentiment.pdf').exists())
            self.assertTrue((p/'figures/trajectories.csv').exists())

if __name__=='__main__':unittest.main()
