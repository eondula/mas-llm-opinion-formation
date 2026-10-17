#!/usr/bin/env bash
# Start an independent dashboard session with all terminal streams disconnected.
set -eu
repository_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_bin="${OPINION_PYTHON:-$repository_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo 'Set OPINION_PYTHON to the Python executable in your CUDA-enabled environment.' >&2
  exit 1
fi
"$python_bin" - "${1:-8766}" <<'PY'
import subprocess
import sys
import time
from pathlib import Path

root=Path.cwd()
log=root/'results/dashboard.log'
log.parent.mkdir(parents=True,exist_ok=True)
with log.open('a') as stream:
    process=subprocess.Popen([sys.executable,'-u','-m','llm_mas','dashboard','--port',sys.argv[1]],cwd=root,
                             stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                             start_new_session=True,close_fds=True)
time.sleep(1)
if process.poll() is not None:
    raise SystemExit('Dashboard failed to start; inspect '+str(log)+' (the port may already be in use).')
print('Dashboard started (PID '+str(process.pid)+'). Log: '+str(log))
print('Dashboard and new workers are independent of this terminal. GPU allocations must remain active.')
PY
