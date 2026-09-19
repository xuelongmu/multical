"""Session-local evaluation cache, bound to calibration and capture contents."""
import hashlib
import json
import os
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def geometry_key(result):
    if result is None:
        return None
    return fingerprint({k: result[k] for k in ('cameras', 'camera_poses')})


def capture_key(records, result):
    ids = set(result.get('training_ids', [])) | set(result.get('validation_ids', []))
    selected = [r for r in records if r['id'] in ids]
    if {r['id'] for r in selected} != ids:
        raise ValueError('Evaluation refers to missing captures')
    return fingerprint(selected)


def save_evaluation(directory, result, baseline, records):
    directory = Path(directory)
    state = dict(version=1, baseline=geometry_key(baseline),
                 board=hashlib.sha256((directory / 'board.yaml').read_bytes()).hexdigest(),
                 captures=capture_key(records, result), result=result)
    path = directory / 'evaluation.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False))
    os.replace(temporary, path)
    return path


def restore_evaluation(directory, baseline, records):
    directory = Path(directory)
    path = directory / 'evaluation.json'
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    result = state['result']
    if (state.get('version') != 1 or state['baseline'] != geometry_key(baseline)
            or state['board'] != hashlib.sha256((directory / 'board.yaml').read_bytes()).hexdigest()
            or state['captures'] != capture_key(records, result)):
        raise ValueError('Saved evaluation does not match this calibration, board or captures')
    return result
