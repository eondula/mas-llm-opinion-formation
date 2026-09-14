import unittest
from llm_mas.config import resolve
from llm_mas.engine import conversation
from llm_mas.prompts import templates, for_condition
from test_combined_reasoning import GeneratingFake

class DiscussionTests(unittest.TestCase):
    def test_named_agents_and_separate_generation_instruction(self):
        for style in ('open_discussion',):
            c=resolve(dict(argument_mode='combined_reasoning',prompt_style=style,rounds=1))
            f=GeneratingFake()
            rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:f})
            original=for_condition(templates(),dict(c,prompt_style='original'),c['scenarios'][0])['stance_probe']
            self.assertEqual(for_condition(templates(),c,c['scenarios'][0])['stance_probe'],original)
            for row,(messages,_,prefix) in zip(rows,f.generated):
                self.assertIn('Your name is '+row['agent'],messages[0]['content'])
                self.assertNotEqual(messages[-1]['content'],row['decision']['probe'])
                self.assertEqual(row['message'],prefix+row['argument'])
                if row['round']==0:self.assertIn('no peer has spoken',messages[-1]['content'])
                else:self.assertIn('Respond to the peer messages',messages[-1]['content'])
            self.assertIn('Agent A said:',str(rows[-1]['decision_history']))

    def test_reject_incompatible_mode(self):
        with self.assertRaises(ValueError):resolve(dict(prompt_style='open_discussion',argument_mode='predefined'))

    def test_topic_has_options_without_supplied_context_or_arguments(self):
        from llm_mas.prompts import initial
        c=resolve(dict(argument_mode='combined_reasoning',prompt_style='topic_discussion',scenarios=['sensor_contextual'],rounds=1))
        s=c['scenarios'][0]
        f=GeneratingFake()
        rows=conversation(c,s,templates(),{c['models'][0]:f})
        for row,(messages,_,_) in zip(rows,f.generated):
            joined=str(messages)
            self.assertIn(s['topic'],joined)
            self.assertIn('Option A:',joined)
            self.assertIn('Option B:',joined)
            self.assertNotIn(s['context'],joined)
            for k in (1,2):
                self.assertIn(s[f'strategy_{k}']['candidate'],joined)
                self.assertNotIn(s[f'strategy_{k}']['property_1'],joined)
                for argument in s[f'strategy_{k}']['arguments']:
                    self.assertNotIn(argument,joined)
                    self.assertNotIn(argument,row['decision']['probe'])
        reversed_probe=for_condition(templates(),dict(c,reverse_candidate_blocks=True),s)['stance_probe']
        self.assertLess(reversed_probe.index('STRATEGY 2'),reversed_probe.index('STRATEGY 1'))
