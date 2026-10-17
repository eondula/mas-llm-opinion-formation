"""Restore the local manifest without changing the frozen collection code."""
from pathlib import Path
import hashlib
import json
import shutil


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'studies/pilot_question_selection.json'
    target=root/'docs/pilot_question_selection.json'
    lock=json.loads((root/'studies/pilot_protocol.lock.json').read_text())
    expected=lock['descriptor']['selection_sha256']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
        raise ValueError('Bundled question manifest differs from frozen protocol')
    if target.exists():
        if target.read_bytes()!=source.read_bytes():
            raise ValueError('Existing local question manifest differs; refusing to overwrite it')
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    print('Pilot question manifest ready. Frozen runtime and protocol unchanged.')


if __name__=='__main__':main()
