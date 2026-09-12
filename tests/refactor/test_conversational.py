import unittest
from llm_mas.config import resolve
from llm_mas.engine import conversation
from llm_mas.prompts import templates
from llm_mas.planning import configurations

class Fake:
    def __init__(self):self.calls=[]
    def measure(self,*args):return dict(margin=-1,stance=-.46)
    def generate(self,messages,max_tokens):
        self.calls.append(messages)
        return {'output':f'Reason {len(self.calls)}.'}

class ConversationalTests(unittest.TestCase):
    def test_schedules_and_history(self):
        for schedule in ['synchronous','sequential','reverse_sequential','no_peer']:
            with self.subTest(schedule=schedule):
                c=resolve(dict(rounds=1,argument_mode='conversational',schedule=schedule,initialization=[1,2]))
                f=Fake();rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:f})
                self.assertEqual(len(f.calls),len(rows))
                for row,call in zip(rows,f.calls):
                    instruction=row['generation']['instruction']
                    self.assertEqual(call[-1]['content'],instruction)
                    self.assertEqual(call[:-1],row['decision_history'])
                    self.assertNotIn(instruction,[m['content'] for m in row['history']])
                    self.assertEqual(row['history'][-1]['content'],row['message'])
                    self.assertTrue(row['message'].endswith(row['argument']))
                    self.assertEqual('No peer message has been delivered' in instruction,not bool(row['deliveries']))
                    if row['round']:self.assertEqual(row['strategy'],2)
                    for delivery in row['deliveries']:
                        source=next(r for r in rows if r['message']==delivery['message'])
                        expected=0 if schedule=='synchronous' else (1 if rows.index(source)<rows.index(row) and source['round']==1 else 0)
                        self.assertEqual(source['round'],expected)
    def test_study_condition(self):
        c=resolve(dict(mode='study',conditions=[{'argument_mode':'conversational'}]))
        self.assertEqual(resolve(next(configurations(c)))['argument_mode'],'conversational')
