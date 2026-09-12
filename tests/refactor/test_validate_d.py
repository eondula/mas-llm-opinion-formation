import unittest
from llm_mas.validate_d import probe, choice, similarities


class ValidationDTests(unittest.TestCase):
    def test_order_moves_arguments_with_candidates(self):
        scenario = {f'strategy_{k}':dict(candidate=f'C{k}', arguments=[f'R{k}']) for k in (1,2)}
        templates = {'combined_reasoning':'intro\nSTRATEGY 1\n{candidate_1}\n{arguments_1}\n\nSTRATEGY 2\n{candidate_2}\n{arguments_2}\n\nReturn the exact candidate statement'}
        q = probe(scenario, templates, order=(2,1))
        self.assertLess(q.index('STRATEGY 2'), q.index('STRATEGY 1'))
        self.assertIn('STRATEGY 2\nC2\n- R2', q)
        self.assertIn('STRATEGY 1\nC1\n- R1', q)

    def test_parser_does_not_repair_invalid_first_line(self):
        self.assertEqual(choice('A\nBecause'), 'A')
        self.assertIsNone(choice('A or B'))
        self.assertIsNone(choice('I prefer A'))
        self.assertIsNone(choice(''))

    def test_similarity_excludes_other_agent_and_conversation(self):
        def row(agent, t, text): return dict(agent=agent, round=t, argument=text)
        records = [('one', {'rows':[row('A',0,'x'),row('B',0,'y'),row('A',1,'y'),row('A',2,'x'),row('A',3,'')]}),
                   ('two', {'rows':[row('A',0,'x')]})]
        rows = similarities(records, {'x':[1.,0.], 'y':[0.,1.]})
        self.assertIsNone(rows[0]['similarity'])
        self.assertEqual(rows[2]['similarity'], 0.)
        self.assertEqual(rows[3]['similarity'], 1.)
        self.assertEqual(rows[3]['matched_round'], 0)
        self.assertEqual(rows[3]['adjacent_similarity'], 0.)
        self.assertEqual(rows[4]['status'], 'missing')
        self.assertIsNone(rows[5]['similarity'])
