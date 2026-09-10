"""Preview launch regressions without loading pretrained checkpoints."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_mas.config import resolve


class PreviewStartTests(unittest.TestCase):
    def test_null_optional_conditions(self):
        self.assertEqual(resolve({'conditions': None})['conditions'], [])

    def test_dashboard_config_keeps_preview_mode_in_worker(self):
        from llm_mas.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'preview.json'
            config.write_text(json.dumps({'mode': 'preview', 'rounds': 1}))
            with patch('sys.argv', ['llm_mas', 'study', '--config', str(config), '--run-id', 'preview_test']), patch('llm_mas.runner.execute') as execute:
                main()
            self.assertEqual(execute.call_args.args[0]['mode'], 'preview')
            self.assertEqual(execute.call_args.args[1], 'preview_test')

    def test_preview_does_not_parse_stale_study_conditions(self):
        script = Path('dashboard/study.js').read_text()
        function = script.split('function config(){', 1)[1].split('\nasync function load', 1)[0]
        javascript = '''
const assert=require('node:assert/strict');
const fields={mode:{value:'preview'},'paper-suite':{checked:true},'run-id':{value:''},
agents:{value:'2'},models:{value:'Qwen/Qwen2.5-1.5B-Instruct'},
scenarios:{selectedOptions:[{value:'storage'}]},rounds:{value:'1'},
initialization:{value:'natural'},schedule:{value:'synchronous'},
argument_mode:{value:'predefined'},max_argument_tokens:{value:'128'},
conditions:{value:'invalid study JSON'}};
const $=id=>fields[id];
''' + 'function config(){' + function + '''
const result=config();
assert.equal(result.mode,'preview');assert.equal(result.paper_suite,false);
assert.deepEqual(result.conditions,[]);assert.deepEqual(result.research,[]);
'''
        subprocess.run(['node', '-e', javascript], check=True)
