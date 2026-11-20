#!/usr/bin/env python3
"""Freeze Q1 development choices before any test outcomes are read.

AI-assisted by Codex, 2026-11-20. No test data are loaded by this script.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json


def main():
    out=Path('artifacts/c_q1_v2');target=out/'freeze.json'
    if target.exists():raise RuntimeError('Freeze already exists; never overwrite after test evaluation')
    candidates={};priority={'frequency':0,'transition':1,'duration':2,'motion_no_flags':3,'motion':4,'geometry':5,'head_video':6,'all_video':7}
    for tag in ['numerical','video5','video15']:
        folder=out/('quality_'+tag);assert (folder/'run_manifest.json').exists()
        for h,models in json.loads((folder/'models.json').read_text()).items():
            for name,spec in models.items():
                candidates.setdefault(h,[]).append(dict(tag=tag,method=name,spec=spec,model_path=str(folder/f'model_{float(h):.1f}_{name}.joblib'),priority=priority[name]))
    selected={}
    # Quantification of the plan's "prefer simpler on tiny differences", fixed
    # before 15 Hz evaluation: within 1% relative Brier, choose fewer inputs.
    for h,rows in candidates.items():
        best=min(r['spec']['score'] for r in rows);eligible=[r for r in rows if r['spec']['score']<=best*1.01]
        selected[h]=min(eligible,key=lambda r:(r['priority'],r['spec']['score']))
    paths=[Path('configs/c_q1_v2.json'),out/'state_models_development.json',out/'q2_sparse_calibration.json']
    paths+=list(Path('src/shuwei_c').glob('*.py'))
    paths+=[p for p in Path('scripts').glob('*c_q1_v2*.py') if not p.name.startswith('report_')]
    for tag in ['numerical','video5','video15']:
        paths+=list((out/('quality_'+tag)).glob('*.joblib'));paths+=[out/('quality_'+tag)/'models.json']
    freeze=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),test_used_before_freeze=False,
        selection_rule='state: lowest validation normalized mean error; quality: simplest within 1% of minimum validation any-hand Brier',
        state=json.loads((out/'state_models_development.json').read_text()),quality=selected,quality_candidates=candidates,
        limitations=['quality model selection uses only four validation environments','physical truth remains unverified','sparse correction applies only to declared probe'],
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    target.write_text(json.dumps(freeze,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(frozen_at=freeze['timestamp_utc'],quality={h:dict(tag=s['tag'],method=s['method']) for h,s in selected.items()})))


if __name__=='__main__':main()
