#!/usr/bin/env python3
"""History ablation, paired cluster intervals, video coverage and source checks.

AI-assisted by Codex, 2026-11-20; development data only.
"""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset,balanced_weights
from shuwei_c.q1v2_state import linear_moments,selection_score
from run_c_q1_v2_state import evaluate


def main():
    out=Path('artifacts/c_q1_v2');config=json.loads(Path('configs/c_q1_v2.json').read_text());model=json.loads((out/'state_models_development.json').read_text());data=Dataset();train=data.records('train');val=data.records('validation')
    moments=linear_moments(train,.1,config,model['noise']);w=balanced_weights([r['environment'] for r in moments],[r['episode'] for r in moments])
    g=np.einsum('n,nij->ij',w,np.array([r['gram'] for r in moments]));b=np.einsum('n,ni->i',w,np.array([r['target'] for r in moments]));penalty=model['horizons']['0.1']['selected']['penalty'];rows=[]
    for k in [1,2,3,4]:
        coef=np.zeros(4);prior=np.zeros(k);prior[0]=1.
        coef[:k]=np.linalg.solve(g[:k,:k]+penalty*np.eye(k),b[:k]+penalty*prior)
        f=evaluate(val,.1,config,dict(method='multiscale',coefficients=coef.tolist()),model['noise'])
        rows.append(dict(lags=k,longest_lag_s=config['lags_s'][k-1],penalty=penalty,coefficients=json.dumps(coef.tolist()),normalized_error=selection_score(f)))
    pd.DataFrame(rows).to_csv(out/'history_ablation.csv',index=False)
    f=pd.read_parquet(out/'state_validation_0.100000.parquet');f['normalized']=f.error/f.scale
    record=f.groupby(['method','dataset','environment','episode','hand']).normalized.mean().groupby(['method','dataset','environment','episode']).mean().unstack('method').reset_index();record.to_csv(out/'state_paired_records.csv',index=False)
    paired=[];rng=np.random.default_rng(config['seed'])
    for ds in ['EgoDemo','EgoPro','EgoStandard','ALL']:
        g=record if ds=='ALL' else record[record.dataset==ds]
        env=g.groupby('environment')[['multiscale','scalar_damped','linear']].mean()
        for base in ['linear','scalar_damped']:
            gain=1-env.multiscale.mean()/env[base].mean();draw=[]
            for _ in range(5000):
                r=env.iloc[rng.integers(len(env),size=len(env))];draw.append(1-r.multiscale.mean()/r[base].mean())
            paired.append(dict(dataset=ds,baseline=base,gain=gain,low=np.quantile(draw,.025),high=np.quantile(draw,.975),
                environments=len(env),records=len(g),records_improved=int((g.multiscale<g[base]).sum()),
                interpretation='paired environment bootstrap; conditional on model selection; one-environment intervals are degenerate'))
    pd.DataFrame(paired).to_csv(out/'state_paired_gains.csv',index=False)
    video=[];dimensions=[]
    for root in ['artifacts/c_video','artifacts/c_video_v2_15hz']:
        manifest=json.loads((Path(root)/'video_manifest.json').read_text())
        for item in manifest:
            raw=[json.loads(line) for line in (Path(root)/item['features_file']).open()]
            video.append(dict(rate=item['sample_rate_hz'],dataset=item['dataset'],split=item['split'],view=item['view'],environment=item['environment'],
                episode=item['episode'],frames=len(raw),any_detected=sum(bool(r['detections']) for r in raw),both_detected=sum(len(r['detections'])==2 for r in raw)))
            if item['dataset']=='EgoDemo':
                rec=data.record(item['episode']);meta=pd.read_parquet(rec.folder/'meta/episodes/chunk-000/file-000.parquet');k=np.array(meta['camera_intrinsics/'+item['view']].iloc[0])
                centered=all(abs(2*k[2]-r['width'])<1e-8 and abs(2*k[3]-r['height'])<1e-8 for r in raw)
                # Only the head-camera feature code uses the centered assumption.
                # Wrist calibration correctly contains noncentral principal points.
                if item['view'].startswith('head_'):assert centered
                dimensions.append(dict(episode=item['episode'],view=item['view'],rate=item['sample_rate_hz'],width=raw[0]['width'],height=raw[0]['height'],principal_point_center_verified=centered))
    pd.DataFrame(video).to_csv(out/'video_detection_coverage.csv',index=False);pd.DataFrame(dimensions).to_csv(out/'demo_camera_dimensions.csv',index=False)
    print('Ablation, paired intervals and video dimensions verified')


if __name__=='__main__':main()
