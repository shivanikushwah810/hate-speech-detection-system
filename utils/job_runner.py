"""
utils/job_runner.py
===================
Web page par "Run" button dabane par evaluate_model.py / train_and_compare.py
ko BACKGROUND process mein chalata hai (taaki browser hang na ho), aur page
status poll karta hai.

Security: only these fixed scripts with fixed arguments can run; the browser sends no arguments.
"""

import os
import re
import subprocess
import sys
import threading
import time

from utils.ml_common import BASE_DIR, LOGS_DIR

# name -> (script, fixed arguments)
JOBS = {
    'evaluation': ('evaluate_model.py', []),
    'comparison': ('train_and_compare.py', []),
    'dataset': ('download_dataset.py', ['--force']),
}

_lock = threading.Lock()
_state = {}   # name -> {'proc', 'started', 'finished', 'returncode'}


def _log_path(name):
    return os.path.join(LOGS_DIR, f'{name}.log')


def _tail(path, max_chars=4000):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_chars))
            text = f.read().decode('utf-8', errors='replace')
    except OSError:
        return ''
    # hide noisy library warnings; keep our progress lines
    noisy = re.compile(r'(Warning:|warnings\.warn|^https?://)')
    lines = [ln for ln in text.splitlines() if not noisy.search(ln)]
    return '\n'.join(lines[-25:])


def start(name):
    """Start a job. Returns (ok, message)."""
    if name not in JOBS:
        return False, 'Unknown job.'
    with _lock:
        st = _state.get(name)
        if st and st['proc'].poll() is None:
            return False, 'This job is already running.'
        # one heavy job at a time
        for other, ost in _state.items():
            if other != name and ost['proc'].poll() is None:
                return False, f'Another job ({other}) is running. Wait for it to finish.'
        os.makedirs(LOGS_DIR, exist_ok=True)
        log = open(_log_path(name), 'wb')
        try:
            script, args = JOBS[name]
            proc = subprocess.Popen([sys.executable, os.path.join(BASE_DIR, script), *args],
                                    cwd=BASE_DIR, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL)
        except OSError as e:
            log.close()
            return False, f'Could not start the job: {e}'
        _state[name] = {'proc': proc, 'started': time.time(), 'finished': None,
                        'returncode': None, 'log': log}
    return True, 'Started.'


def status(name):
    if name not in JOBS:
        return {'state': 'unknown'}
    with _lock:
        st = _state.get(name)
        if not st:
            return {'state': 'idle', 'log': _tail(_log_path(name)) if os.path.exists(_log_path(name)) else ''}
        rc = st['proc'].poll()
        if rc is not None and st['finished'] is None:
            st['finished'] = time.time()
            st['returncode'] = rc
            try:
                st['log'].close()
            except Exception:
                pass
        state = 'running' if rc is None else ('finished' if rc == 0 else 'failed')
        end = st['finished'] or time.time()
        return {'state': state, 'returncode': rc, 'seconds': round(end - st['started'], 1),
                'log': _tail(_log_path(name))}
