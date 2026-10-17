"""Reconnect dashboards to owned detached workers using PID and creation time."""
import json
import os
import signal
import socket
import time
from pathlib import Path
import psutil
from .storage import write

TERMINAL={'complete','failed','cancelled'}


def register(folder,process,command):
    # Popen mocks and already-exited startup failures have no live identity.
    if type(process.pid) is not int:return
    try:created=psutil.Process(process.pid).create_time()
    except psutil.Error:return
    write(Path(folder)/'worker.json',dict(pid=process.pid,created=created,command=command,started_at=time.time(),host=socket.gethostname()))


def worker(folder):
    folder=Path(folder)
    try:
        info=json.loads((folder/'worker.json').read_text())
        if info.get('host',socket.gethostname())!=socket.gethostname():return None
        command=info['command'];i=command.index('-m')
        if command[i+1:i+3]!=['llm_mas','run']:return None
        if command[command.index('--run-id')+1]!=folder.name:return None
        expected=folder.parent/'configurations'/(folder.name+'.json')
        if Path(command[command.index('--config')+1]).resolve()!=expected.resolve():return None
        process=psutil.Process(info['pid'])
        if abs(process.create_time()-info['created'])>.01:return None
        if hasattr(os,'getuid') and process.uids().real!=os.getuid():return None
        if process.status() in (psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):return None
        if process.cmdline()[1:]!=command[1:]:return None
        return process
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,psutil.Error):return None


def running(root):
    names=[]
    for folder in (Path(root)/'results/experiments').glob('*'):
        if not (folder/'config.json').exists():continue
        try:status=json.loads((folder/'status.json').read_text())
        except (OSError,ValueError):status={}
        if status.get('phase') not in TERMINAL and worker(folder) is not None:names.append(folder.name)
    return names


def latest(root,active=()):
    choices=[]
    for folder in (Path(root)/'results/experiments').glob('*'):
        config=folder/'config.json'
        if not config.is_file():continue
        try:
            started=json.loads((folder/'worker.json').read_text()).get('started_at',config.stat().st_mtime)
        except (OSError,ValueError,TypeError,AttributeError):started=config.stat().st_mtime
        if type(started) not in (int,float):started=config.stat().st_mtime
        choices.append((folder.name in active,started,folder.name))
    return max(choices)[2] if choices else None


def stop(folder):
    process=worker(folder)
    if process is None:return False
    try:process.send_signal(signal.SIGINT)
    except psutil.Error:return False
    return True


def worker_host(folder):
    try:return json.loads((Path(folder)/'worker.json').read_text()).get('host')
    except (OSError,ValueError,AttributeError):return None
