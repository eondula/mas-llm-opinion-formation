import unittest
from llm_mas.config import scenario, validate_scenario
from llm_mas.prompts import initial, templates

class ContextTests(unittest.TestCase):
    def test_context_is_rendered_in_both_orders_and_styles(self):
        for name in ('sensor','storage'):
            s=scenario(name+'_contextual')
            for order in ('strategy_1_first','strategy_2_first'):
                for style in ('original','open_discussion'):
                    messages=initial(s,templates(),order,prompt_style=style)
                    self.assertIn(s['context'],messages[1]['content'])
                    first=s['strategy_1' if order=='strategy_1_first' else 'strategy_2']['property_1']
                    self.assertIn('Option A:\n- '+first,messages[1]['content'])
            plain=dict(s);plain.pop('context')
            self.assertEqual(initial(s,templates())[1]['content'],s['context']+'\n\n'+initial(plain,templates())[1]['content'])
            s['context']=123
            with self.assertRaises(ValueError):validate_scenario(s)
