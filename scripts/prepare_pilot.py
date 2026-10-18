"""Verify tracked pilot inputs. No docs folder or manual copying is required."""
from pathlib import Path
import json


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'studies/pilot_question_selection.json'
    selection=json.loads(source.read_text())
    for device in ('mps','cuda'):
        config=json.loads((root/f'studies/pilot_{device}_v6.json').read_text())
        if config['question_ids']!=selection['question_ids']:
            raise ValueError('Pilot selection differs from bundled manifest')
    print('Tracked pilot inputs ready. No docs folder is required.')


if __name__=='__main__':main()
