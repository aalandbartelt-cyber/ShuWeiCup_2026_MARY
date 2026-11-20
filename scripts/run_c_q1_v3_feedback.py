#!/usr/bin/env python3
"""Compare delayed empirical coverage feedback and nested radius levels."""
from pathlib import Path
import sys,json,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q1v3_feedback import adapt_radii
from shuwei_c.q1v3_calibration import radius_metrics


def main():
    source=Path('artifacts/c_q1_v3/calibration_screen');out=Path('artifacts/c_q1_v3/feedback_screen');out.mkdir(exist_ok=True);start=time.perf_counter()
    frame=pd.read_parquet(source/'validation_features.parquet');a=np.load(source/'validation_radii.npz');levels=[.5,.8,.9,.95]
    base=np.stack([a[f'quantile_online_leaves15_dataset_{level}'] for level in levels],axis=1);rows=[];radii={};groups=[]
    crossing=float(np.mean(np.any(np.diff(base,axis=1)<0,axis=1)))
    for rate in [0.,.005,.01,.02,.05,.1]:
        r,m=adapt_radii(frame,base,levels,rate);radii[str(rate)]=r
        for k,level in enumerate(levels):
            values=radius_metrics(frame,r[:,k],level,f'feedback_{rate}')
            for v in values:v['rate']=rate
            rows+=values
        for env,g in frame.groupby('environment'):
            values=radius_metrics(g.reset_index(drop=True),r[g.index,2],.9,f'feedback_{rate}')
            for v in values:v.update(environment=env,rate=rate)
            groups+=values
    pd.DataFrame(rows).to_csv(out/'validation_metrics.csv',index=False);pd.DataFrame(groups).to_csv(out/'environment_metrics.csv',index=False);np.savez_compressed(out/'validation_radii.npz',**radii)
    (out/'protocol.json').write_text(json.dumps(dict(test_used=False,levels=levels,raw_crossing_fraction=crossing,rate_units='per 1/30 second of qualified arriving reference time',interpretation='Development screen; feedback only after qualified reference arrives; nested levels by cumulative maximum; no arbitrary missing-data guarantee',elapsed_s=time.perf_counter()-start,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_feedback.py')]}),indent=2)+'\n')


if __name__=='__main__':main()
