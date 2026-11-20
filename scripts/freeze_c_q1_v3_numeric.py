#!/usr/bin/env python3
"""Freeze v3 point/radius components before fixed-version test reevaluation.

The original v2 test was already observed. This is not a fresh unseen test.
Quality/video candidates remain a separate train/validation development track.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib


def main():
    root=Path('artifacts/c_q1_v3');out=root/'frozen_numeric';out.mkdir(exist_ok=True);target=out/'freeze.json'
    if target.exists():raise RuntimeError('Numerical component freeze already exists')
    spec=json.loads((root/'interface/development_specification.json').read_text());spec['version']='q1-v3-numeric-fixed-20261120'
    spec['timestamp_utc']=datetime.now(timezone.utc).isoformat();spec['evaluation_status']='The v2 official test has already been observed; subsequent evaluation is explicitly a fixed-version reevaluation'
    spec['scope']='Point models at five declared horizons; conditional ball and delayed feedback at 0.1 s only. Quality/video selection is separate and cannot modify these fixed numerical components.'
    paths=[Path('configs/c_q1_v2.json'),Path('scripts/evaluate_c_q1_v3_numeric.py'),Path(__file__)]
    for name in ['q1v3_state.py','q1v3_learning.py','q1v3_calibration.py','q1v3_feedback.py']:paths.append(Path('src/shuwei_c')/name)
    for checkpoints in spec['point_checkpoints'].values():paths.extend(map(Path,checkpoints))
    paths.extend(map(Path,spec['calibration']['quantile_models'].values()));paths.extend([Path(spec['calibration']['parameters']),Path(spec['calibration']['flagged_floor_parameters'])])
    spec['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old=json.loads(Path('artifacts/c_q1_v2/freeze.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in old['source_hashes'].items())
    spec['inherited_v2_hashes']=old['source_hashes'];target.write_text(json.dumps(spec,indent=2,ensure_ascii=False)+'\n')
    for p in paths:
        if p.suffix=='.py':
            archive=out/'source_snapshot'/p;archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(p.read_bytes())
    print(json.dumps(dict(timestamp_utc=spec['timestamp_utc'],components=len(spec['point_checkpoints']),test_previously_observed=True)))


if __name__=='__main__':main()
