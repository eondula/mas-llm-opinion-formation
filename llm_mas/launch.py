"""Explicit one-use handoff from dashboard reservation to its worker."""
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from .storage import read,write
from .config import digest

STARTUP_FILES={'config.json','status.json','worker.log','worker.json','launch.json','error.log'}


def startup_entries(path):
    """Exclude only known atomic metadata temporaries and NFS unlink placeholders."""
    entries=[]
    for child in Path(path).iterdir():
        if child.name in STARTUP_FILES and child.is_file():continue
        if child.name.endswith('.tmp') and child.name[:-4] in STARTUP_FILES and child.is_file():continue
        if re.fullmatch(r'\.nfs[0-9a-fA-F]+',child.name) and child.is_file():continue
        entries.append(child.name)
    return sorted(entries)


def reserve(folder,config):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    token=secrets.token_hex(32)
    write(folder/'config.json',config)
    write(folder/'launch.json',dict(version=1,run_id=folder.name,config_hash=digest(config),token_sha256=hashlib.sha256(token.encode()).hexdigest()))
    return token


def resolved_reservation_config(saved):
    """An older dashboard may omit defaults supplied by the current worker."""
    from .config import resolve
    normalized=resolve(saved)
    normalized.pop('run_id',None)
    return normalized


def legacy_reservation(folder,config):
    folder=Path(folder)
    if not folder.exists() or startup_entries(folder) or not (folder/'config.json').is_file():return False
    saved=read(folder/'config.json')
    if saved.get('run_id') not in (None,folder.name):return False
    return resolved_reservation_config(saved)==config


def claim(folder,config,token):
    folder=Path(folder)
    if not isinstance(token,str) or not re.fullmatch('[0-9a-f]{64}',token):raise ValueError('Invalid dashboard launch token')
    marker=folder/'launch.json'
    if not marker.is_file():raise ValueError('Dashboard launch reservation is missing; use New experiment with a new ID')
    launch=read(marker)
    if launch.get('run_id')!=folder.name or not secrets.compare_digest(launch.get('token_sha256',''),hashlib.sha256(token.encode()).hexdigest()):
        raise ValueError('Dashboard launch reservation does not belong to this worker')
    unexpected=startup_entries(folder)
    if unexpected:raise ValueError('Experiment contains existing data or a claimed launch: '+', '.join(unexpected)+'; choose a new experiment ID')
    saved=read(folder/'config.json')
    if digest(saved)!=launch.get('config_hash'):raise ValueError('Dashboard reservation configuration was modified')
    canonical=resolved_reservation_config(saved)
    if canonical!=config:
        keys=sorted(k for k in canonical.keys()|config.keys() if canonical.get(k)!=config.get(k))
        raise ValueError('Dashboard/worker configuration differs in: '+', '.join(keys)+'; restart the dashboard after updating code')
    # Atomic exclusive creation makes the handoff single-use, including on shared filesystems.
    with (folder/'launch.claimed').open('x') as stream:
        json.dump(dict(pid=os.getpid(),config_hash=digest(config)),stream)
    return True
