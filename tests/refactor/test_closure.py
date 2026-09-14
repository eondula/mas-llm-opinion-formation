import json
import unittest
from llm_mas.closure import check
from llm_mas.config import resolve
from llm_mas.engine import conversation
from llm_mas.prompts import templates

class Fake:
    def __init__(self, signals): self.signals=iter(signals); self.interaction_context={}; self.calls=[]
    def generate(self, history, tokens):
        self.calls.append(history)
        action=next(self.signals)
        return dict(output=json.dumps(dict(conversation_action=action, reason='My reason')) if action else 'invalid')
    def measure(self,*args):return dict(margin=1,stance=.4)

class ClosureTests(unittest.TestCase):
    def run_check(self, signals, **overrides):
        c=resolve(dict(mode='joint_task',stopping='agent_closure',rounds=2,**overrides))
        f=Fake(signals);h=[[dict(role='user',content='task')] for _ in range(2)]
        rows=[dict(agent='Agent A',strategy=1,message='A'),dict(agent='Agent B',strategy=2,message='B')]
        result=check(c,templates(),{'x':f},['x','x'],['Agent A','Agent B'],h,rows,1)
        return result,f,h
    def test_same_proposal_confirmation_can_close_in_disagreement(self):
        result,f,h=self.run_check(['finish','finish','finish','finish'])
        self.assertEqual(result['reason'],'agents_accepted_closure')
        self.assertEqual(f.calls[2][-2]['content'],f.calls[3][-2]['content'])
        self.assertIn('Closure responses',h[0][-1]['content'])
    def test_initial_finish_signals_do_not_count_as_confirmation(self):
        for vote in ('continue',None):
            result,_,_=self.run_check(['finish','finish','finish',vote])
            self.assertIsNone(result['reason'])
    def test_no_proposal_no_confirmation(self):
        result,f,_=self.run_check(['continue','continue'])
        self.assertIsNone(result['proposal']);self.assertEqual(len(f.calls),2)
    def test_no_peer_does_not_exchange_or_generate_controls(self):
        result,f,h=self.run_check([],schedule='no_peer')
        self.assertEqual(len(f.calls),0);self.assertEqual(len(h[0]),1)
        self.assertIn('No peer',result['unavailable'])
    def test_engine_stops_and_budget_is_distinct(self):
        for signals,expected in [(['finish']*4,'agents_accepted_closure'),(['continue']*4,'budget_exhausted')]:
            c=resolve(dict(mode='joint_task',stopping='agent_closure',rounds=2))
            f=Fake(signals)
            rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:f})
            self.assertEqual(rows[-1]['termination']['reason'],expected)
            self.assertEqual(rows[-1]['round'],1 if expected=='agents_accepted_closure' else 2)
            self.assertIn('history_after_closure',rows[-1])
