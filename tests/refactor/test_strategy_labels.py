import unittest
from llm_mas.config import scenario
from llm_mas.prompts import initial, templates

class StrategyLabelTests(unittest.TestCase):
    def test_labels_follow_identity_in_both_orders_and_all_styles(self):
        texts=templates()
        for task in ('sensor','storage','sensor_contextual','storage_contextual'):
            s=scenario(task)
            for style in ('original','open_discussion','topic_discussion','discussion_then_decision'):
                for order in ('strategy_1_first','strategy_2_first'):
                    with self.subTest(task=task,style=style,order=order):
                        content=initial(s,texts,order,prompt_style=style)[1]['content']
                        for k in (1,2):
                            expected=(f'Strategy {k}: '+s[f'strategy_{k}']['candidate'] if style=='topic_discussion' else
                                      f'Strategy {k}:\n- '+s[f'strategy_{k}']['property_1']+'\n- '+s[f'strategy_{k}']['property_2'])
                            self.assertIn(expected,content)
                        first,second=(2,1) if order=='strategy_2_first' else (1,2)
                        self.assertLess(content.index(f'Strategy {first}:'),content.index(f'Strategy {second}:'))
                        self.assertNotIn('Option A:',content)
                        self.assertNotIn('Option B:',content)
