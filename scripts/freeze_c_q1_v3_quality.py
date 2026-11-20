#!/usr/bin/env python3
"""Freeze endpoint risk and video-assisted onset risk before test extraction."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from shuwei_c import q1v2_quality as q


def main():
    root=Path('artifacts/c_q1_v3');out=root/'frozen_quality';out.mkdir(exist_ok=True)
    if (out/'freeze.json').exists():raise RuntimeError('Quality component freeze already exists')
    parity=json.loads((root/'head_diagnostic_adapter/verification.json').read_text());assert parity['status']=='PASS' and parity['probability_max_difference']==0
    seeds=[20261120,20261121,20261122];endpoint={};paths=[]
    for h,reg,folder in [(.1,.01,'articulated_oof'),(.5,.001,'articulated_oof_h0.5_frequency')]:
        frame=pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_train_{h}.parquet');base=q.fit_baseline(frame,h==.1);path=out/f'endpoint_prior_{h}.joblib';joblib.dump(base,path);paths.append(path)
        checkpoints=[str(root/'articulated_risk'/f'numeric_r{reg}_s{s}_e20.pt') if h==.1 else str(root/folder/f'full_numeric_r{reg}_s{s}.pt') for s in seeds]
        cache=pd.read_parquet(root/folder/'train_oof.parquet');prob=cache[[f'numeric_r{reg}_p{k}' for k in range(4)]].to_numpy()
        thresholds={mode:{str(level):q.select_threshold(prob,frame,level,mode) for level in [.01,.05,.1]} for mode in ['onset','recovery']}
        endpoint[str(h)]=dict(horizon=h,regularization=reg,checkpoints=checkpoints,prior_path=str(path),thresholds=thresholds,feature_recipe='q1v3_shared_risk.features(video=False), followed by per-hand current flag')
        paths.extend(map(Path,checkpoints))
    source=root/'onset_h0.5';meta=json.loads((source/'manifest.json').read_text());events=pd.read_csv(source/'events.csv');models={}
    for name in ['frequency','numeric_leaves7','diagnostics_leaves7']:
        path=source/('frequency.joblib' if name=='frequency' else f'full_{name}.joblib');paths.append(path)
        models[name]=dict(path=str(path),thresholds={str(float(r.nominal_false_alarm)):float(r.threshold) for r in events[events.method==name].itertuples()})
    originals=json.loads(Path('artifacts/c_video_v2_15hz/video_manifest.json').read_text());weight=Path('artifacts/c_video_v2_15hz/hand_landmarker.task');paths.append(weight)
    source_paths=['scripts/freeze_c_q1_v3_quality.py','scripts/evaluate_c_q1_v3_quality.py','scripts/extract_c_q1_v3_test_detections.py','scripts/extract_c_q1_v3_test_diagnostics.py','scripts/extract_c_q1_v3_visual.py','scripts/prepare_c_video.py','scripts/run_c_q1_v3_onset.py','src/shuwei_c/q1v3_head_diagnostics.py','src/shuwei_c/q1v3_visual.py','src/shuwei_c/q1v3_shared_risk.py','src/shuwei_c/q1v3_articulated_risk.py']
    paths.extend(map(Path,source_paths));old=json.loads(Path('artifacts/c_q1_v2/freeze.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in old['source_hashes'].items())
    spec=dict(version='q1-v3-quality-fixed-20261120',timestamp_utc=datetime.now(timezone.utc).isoformat(),test_previously_observed=True,
        endpoint=endpoint,onset=dict(horizon=.5,target=meta['target'],numeric_columns=meta['numeric_columns'],models=models,selected='diagnostics_leaves7'),
        visual=dict(detector_weight=str(weight),mediapipe=originals[0]['mediapipe'],views=['head_left','head_right'],rate_hz=15,detector_width=960,flow_width=320,detector_mode='IMAGE',features='Same development crop, projection assignment and previous-to-current Farneback diagnostics; no test fitting or PCA'),
        adapter_validation=parity,inherited_v2_hashes=old['source_hashes'],source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'freeze.json').write_text(json.dumps(spec,indent=2)+'\n')
    for p in map(Path,source_paths):
        target=out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    print(json.dumps(dict(timestamp_utc=spec['timestamp_utc'],onset=spec['onset']['selected'],endpoint=list(endpoint),test_previously_observed=True)))


if __name__=='__main__':main()
