import json
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_mas.config import resolve
from llm_mas.engine import conversation
from llm_mas.joint_task import decision_sheet
from llm_mas.prompts import initial, templates
from test_discussion_then_decision import Runtime


class JointTaskTests(unittest.TestCase):
    def test_prompt_styles_keep_task_and_names(self):
        for style in ('original', 'open_discussion', 'topic_discussion', 'discussion_then_decision'):
            c = resolve(dict(mode='joint_task', prompt_style=style, argument_mode='combined_reasoning', rounds=1))
            rows = conversation(c, c['scenarios'][0], templates(), {c['models'][0]:Runtime()})
            system = rows[0]['history'][0]['content']
            self.assertIn(c['joint_task_brief'], system)
            self.assertIn('Your name is Agent A', system)
            self.assertEqual(rows[-1]['joint_outcome']['status'], 'shared_choice')
            self.assertEqual(len(rows[-1]['joint_outcome']['positions']), 2)

    def test_original_prompt_unchanged_without_task(self):
        c = resolve({})
        messages = initial(c['scenarios'][0], templates())
        self.assertEqual(messages[0]['content'], templates()['system'])
        self.assertNotIn('decision sheet', str(messages))

    def test_unresolved_invalid_and_no_peer(self):
        s = resolve({})['scenarios'][0]
        rows = [dict(round=1, agent='Agent A', strategy=1, argument='Reason A'),
                dict(round=1, agent='Agent B', strategy=2, argument='Reason B')]
        outcome = decision_sheet(rows, s, 2, 'no_peer')
        self.assertEqual(outcome['status'], 'unresolved')
        self.assertIsNone(outcome['recommendation'])
        self.assertFalse(outcome['peer_exchange'])
        rows[-1]['strategy'] = None
        self.assertEqual(decision_sheet(rows, s, 2, 'synchronous')['status'], 'incomplete_or_invalid')

    def test_fixed_arguments_forced_start_and_no_peer_remain_available(self):
        c = resolve(dict(mode='joint_task', initialization=[2,1], schedule='no_peer', rounds=1))
        rows = conversation(c, c['scenarios'][0], templates(), {c['models'][0]:Runtime()})
        self.assertEqual([r['strategy'] for r in rows[:2]], [2,1])
        self.assertTrue(all(not r['deliveries'] for r in rows))
        self.assertFalse(rows[-1]['joint_outcome']['peer_exchange'])

    def test_generated_choices(self):
        c = resolve(dict(mode='joint_task', prompt_style='discussion_then_decision',
                         argument_mode='combined_reasoning', decision_mode='generated', rounds=1))
        rows = conversation(c, c['scenarios'][0], templates(), {c['models'][0]:Runtime()})
        self.assertEqual(rows[-1]['joint_outcome']['strategy'], 2)
        self.assertTrue(all(p['choice_status']=='valid' for p in rows[-1]['joint_outcome']['positions']))

    def test_validation(self):
        for extra in (dict(scenarios=['sensor','storage']), dict(joint_task_brief=''),
                      dict(grid={'schedule':['synchronous']}), dict(research=['validation'])):
            with self.assertRaises(ValueError):
                resolve(dict(mode='joint_task', **extra))
        self.assertEqual(resolve(dict(mode='joint_task', stopping='sign_stability'))['stopping'], 'sign_stability')

    def test_saved_decision_sheet(self):
        from llm_mas.runner import execute
        class Fake(Runtime):
            def __init__(self, *args): pass
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'run'
            with patch('llm_mas.runner.run_path', return_value=path), patch('llm_mas.runtime.Runtime', Fake), patch('llm_mas.figures.render'):
                execute(dict(mode='joint_task', rounds=1, sentiment=False), name='test_joint')
            outcome = json.loads((path/'joint_decision.json').read_text())
            record = json.loads((path/'conversations/000_000.json').read_text())
            self.assertEqual(outcome['positions'], record['joint_outcome']['positions'])
            self.assertEqual(outcome['status'], 'shared_choice')

    def test_dashboard_sheet_renders_disagreement_and_invalid_choice(self):
        script = Path('dashboard/study.js').read_text().split('function drawJointSheet(', 1)[1]
        js = """
const assert=require('node:assert/strict');
const fields={};const $=id=>fields[id]||(fields[id]={children:[],replaceChildren(){this.children=[]},append(p){this.children.push(p)}});
const document={createElement(){return {style:{}}}};
let lastPhase='complete';
let events=[{task:'000_000',row:{round:1,agent:'Agent A',strategy:1,argument:'A'}},{task:'000_000',row:{round:1,agent:'Agent B',strategy:2,argument:'B'}}];
""" + 'function drawJointSheet(' + script + """
drawJointSheet(events,{mode:'joint_task',schedule:'no_peer'},2,'000_000');
assert.match($('joint-decision-status').textContent,/Unresolved disagreement/);
assert.match($('joint-decision-status').textContent,/No peer exchange/);
assert.equal($('joint-decision-positions').children.length,2);
events[1].row.strategy=null;
drawJointSheet(events,{mode:'joint_task'},2,'000_000');
assert.match($('joint-decision-status').textContent,/invalid choice/);
drawJointSheet(events,{mode:'preview'},2,'000_000');
assert.equal($('joint-decision-sheet').hidden,true);
"""
        subprocess.run(['node','-e',js],check=True,capture_output=True,text=True)
