import unittest
from llm_mas.config import resolve
from llm_mas.engine import conversation
from llm_mas.prompts import templates
from llm_mas.runtime import Runtime
from llm_mas.planning import configurations


class SelectingRuntime(Runtime):
    def __init__(self):
        self.calls = []

    def measure(self, history, scenario, texts):
        return dict(margin=1, stance=.46)

    def likelihood(self, messages, candidate):
        self.calls.append((messages, candidate))
        # Choose a non-first entry initially and a different entry after peer delivery.
        peer_seen = any('Agent B' in m['content'] or 'Agent A' in m['content'] for m in messages[:-1])
        wanted = '3' if peer_seen else '2'
        return (0 if candidate == wanted else -4), [], 'rendered selection'


class ArgumentSelectionTests(unittest.TestCase):
    def test_context_selection_and_clean_history(self):
        c = resolve(dict(rounds=1, argument_mode='agent_selected', initialization=[2,1]))
        runtime = SelectingRuntime()
        rows = conversation(c, c['scenarios'][0], templates(), {c['models'][0]:runtime})
        for row, index in zip(rows, [1,1,2,2]):
            pool = c['scenarios'][0][f"strategy_{row['strategy']}"]
            self.assertEqual(row['argument_index'], index)
            self.assertEqual(row['message'], pool['candidate']+'\n'+pool['arguments'][index])
            self.assertEqual(row['argument_selection']['label'], str(index+1))
            self.assertIn(pool['arguments'][-1], row['argument_selection']['prompt'])
            self.assertNotIn('Return only its number', str(row['history']))
            self.assertEqual(len(row['argument_selection']['scores']), len(pool['arguments']))
        self.assertEqual(rows[2]['deliveries'][0]['message'], rows[1]['message'])

    def test_repeats_allowed_and_no_cycling(self):
        c=resolve(dict(rounds=2, argument_mode='agent_selected', schedule='no_peer'))
        rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:SelectingRuntime()})
        self.assertEqual([r['argument_index'] for r in rows],[1]*6)

    def test_fixed_list_conversational_instruction(self):
        c=resolve(dict(rounds=1, argument_mode='agent_selected'))
        runtime=SelectingRuntime()
        rows=conversation(c,c['scenarios'][0],templates(),{c['models'][0]:runtime})
        for row in rows:
            prompt=row['argument_selection']['prompt']
            self.assertIn('latest peer messages',prompt)
            self.assertIn('If no peer messages are present',prompt)
            self.assertIn('Do not rewrite, combine, or invent',prompt)
            self.assertIn('do not change the selected strategy',prompt)
            self.assertIn(row['argument'],c['scenarios'][0][f"strategy_{row['strategy']}"]['arguments'])
        self.assertEqual(len(runtime.calls),sum(len(row['argument_selection']['labels']) for row in rows))

    def test_study_grid_contains_all_modes(self):
        c=resolve(dict(mode='study',grid={'argument_mode':['predefined','agent_selected','generated']}))
        self.assertEqual([resolve(x)['argument_mode'] for x in configurations(c)],
                         ['predefined','agent_selected','generated'])

    def test_nonfinite_selection_rejected(self):
        runtime=SelectingRuntime()
        runtime.likelihood=lambda *args:(float('nan'), [], '')
        c=resolve({})
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            runtime.select_argument([],c['scenarios'][0]['strategy_1'],templates())

if __name__=='__main__':unittest.main()
