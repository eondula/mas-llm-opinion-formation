import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from llm_mas.web import active_runs


class ActiveRunsTest(unittest.TestCase):
    def test_terminal_discovery_excludes_finished_and_unrelated_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,phase in [('live','conversation'),('done','complete')]:
                path=root/name
                path.mkdir()
                (path/'config.json').write_text('{}')
                (path/'status.json').write_text(json.dumps({'phase':phase}))
            processes='\n'.join([
                'python -u -m llm_mas study --config studies/experiment_C.json --run-id live',
                'python -m llm_mas study --run-id done',
                'python -m llm_mas study --run-id missing',
                'python -m other study --run-id live'])
            with patch('llm_mas.web.subprocess.check_output',return_value=processes),patch('llm_mas.web.run_path',side_effect=lambda name:root/name):
                self.assertEqual(active_runs(),['live'])

    def test_process_inspection_unavailable(self):
        with patch('llm_mas.web.subprocess.check_output',side_effect=OSError):
            self.assertEqual(active_runs(),[])
