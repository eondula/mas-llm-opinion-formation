"""Incremental, compact dashboard snapshots; scientific records remain on disk."""
import json
import threading
from collections import OrderedDict
from pathlib import Path


class LiveView:
    def __init__(self, compact, capacity=4):
        self.compact, self.capacity = compact, capacity
        self.runs = OrderedDict()
        self.lock = threading.Lock()

    def snapshot(self, folder):
        with self.lock:
            folder = Path(folder)
            key = str(folder)
            state = self.runs.setdefault(key, dict(offset=0, inode=None, events={}, closures=[], files={}))
            self.runs.move_to_end(key)
            while len(self.runs) > self.capacity:
                self.runs.popitem(last=False)
            log = folder/'events.jsonl'
            if log.exists():
                stat = log.stat()
                if state['inode'] != stat.st_ino or stat.st_size < state['offset']:
                    state.update(offset=0, inode=stat.st_ino, events={}, closures=[], files={})
                with log.open('rb') as stream:
                    stream.seek(state['offset'])
                    for line in stream:
                        # A concurrent append can expose an unfinished JSON line.
                        if not line.endswith(b'\n'):
                            break
                        state['offset'] += len(line)
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            continue
                        if 'row' in event:
                            item = self.compact(event)
                            state['events'][event_key(item)] = item
                        if event.get('kind') == 'termination_check' and event.get('policy') == 'agent_closure':
                            state['closures'].append({k:event.get(k) for k in ('task','round','reason','proposal','unavailable')} | {
                                phase:[{k:v.get(k) for k in ('agent','action','reason','valid')} for v in event.get(phase,[])]
                                for phase in ('proposals','confirmations')})
            # Completed records contain later sentiment/evaluation annotations. Cache
            # only their compact rows, and invalidate after every atomic replacement.
            present = set()
            for file in sorted((folder/'conversations').glob('*.json')):
                present.add(str(file))
                stat = file.stat()
                stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                saved = state['files'].get(str(file))
                if saved is None or saved[0] != stamp:
                    record = json.loads(file.read_text())
                    saved = (stamp, [self.compact(dict(task=file.stem,row=row)) for row in record['rows']])
                    state['files'][str(file)] = saved
            state['files'] = {k:v for k,v in state['files'].items() if k in present}
            events = dict(state['events'])
            for _, rows in state['files'].values():
                for item in rows:
                    events[event_key(item)] = item
            # Stable sorting preserves actual receiver order within each round.
            return sorted(events.values(), key=lambda e:(e['task'],e['row']['round'])), list(state['closures'])


def event_key(event):
    row = event['row']
    return event['task'], row['round'], row['agent']
