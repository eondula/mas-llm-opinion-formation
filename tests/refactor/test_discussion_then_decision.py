import unittest
from llm_mas.config import resolve
from llm_mas.engine import conversation
from llm_mas.prompts import templates


class Runtime:
    def measure(self, history, scenario, texts):
        discussed = history[-1]['role'] == 'assistant' and history[-1]['content'] == 'Discussion marker.'
        return dict(margin=-1 if discussed else 1, stance=0, rendered_prompt='test')

    def generate(self, messages, max_tokens, **kwargs):
        prompt = messages[-1]['content']
        if 'A separate decision step follows' in prompt:
            self.last_discussion_prompt = prompt
            return dict(output='Discussion marker.')
        if 'JSON object' in prompt:
            return dict(output='{"strategy":2,"explanation":"Decision reason."}')
        return dict(output='Initial rationale.')


class DiscussionThenDecisionTests(unittest.TestCase):
    def run_case(self, **overrides):
        config = resolve(dict(argument_mode='combined_reasoning', prompt_style='discussion_then_decision',
                              rounds=1, **overrides))
        runtime = Runtime()
        return conversation(config, config['scenarios'][0], templates(), {config['models'][0]:runtime}), runtime

    def test_scored_reassessment_and_single_retained_discussion(self):
        rows, _ = self.run_case()
        for row in rows[:2]:
            self.assertEqual(row['strategy'], 1)
            self.assertIsNone(row['discussion_generation'])
        for row in rows[2:]:
            self.assertEqual(row['strategy'], 2)
            self.assertEqual(row['pre_discussion']['margin'], 1)
            self.assertEqual(row['decision']['margin'], -1)
            self.assertEqual(row['decision_history'][-1]['content'], 'Discussion marker.')
            self.assertEqual(sum(m['content'].count('Discussion marker.') for m in row['history']), 1)
            self.assertIn('Decision:', row['message'])

    def test_generated_keeps_raw_decision_and_discussion(self):
        rows, _ = self.run_case(decision_mode='generated')
        for row in rows[2:]:
            self.assertEqual(row['choice_status'], 'valid')
            self.assertEqual(row['strategy'], 2)
            self.assertIn('Discussion marker.', row['argument'])
            self.assertEqual(row['generation']['output'], '{"strategy":2,"explanation":"Decision reason."}')
            self.assertIn('Reassess after your discussion', row['generation']['instruction'])

    def test_forced_initialization_and_no_peer(self):
        rows, runtime = self.run_case(initialization=[2,1], schedule='no_peer')
        self.assertEqual([r['strategy'] for r in rows[:2]], [2,1])
        self.assertTrue(all(r['forced'] for r in rows[:2]))
        self.assertIn('do not invent peer statements', runtime.last_discussion_prompt)
        self.assertTrue(all(not r['deliveries'] for r in rows))

    def test_sequential_delivers_completed_decision(self):
        rows, _ = self.run_case(schedule='sequential')
        self.assertEqual(rows[3]['deliveries'][0]['message'], rows[2]['message'])
        self.assertIn('Decision:', rows[3]['deliveries'][0]['message'])
