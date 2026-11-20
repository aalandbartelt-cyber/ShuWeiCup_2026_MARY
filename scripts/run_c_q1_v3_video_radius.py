#!/usr/bin/env python3
"""Visual conditional-radius screening on fixed environment-OOF point errors."""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from shuwei_c.q1v3_state import row_weights
from shuwei_c.q1v3_calibration import conditional_columns,radius_metrics


def main():
    out=Path('artifacts/c_q1_v3/video_radius');out.mkdir(exist_ok=True);started=time.perf_counter()
    frames={s:pd.read_parquet(f'artifacts/c_q1_v3/calibration_screen/{s}_features.parquet') for s in ['train','validation']}
    visual={}
    for split,frame in frames.items():
        meta=pd.read_parquet(f'artifacts/c_q1_v3/state_screen/{split}_0.1_meta.parquet');meta['visual_index']=np.arange(len(meta));keys=['episode','origin','hand']
        ix=frame[keys].merge(meta[keys+['visual_index']],on=keys,validate='one_to_one').visual_index.to_numpy()
        a=np.load(f'artifacts/c_q1_v3/visual_screen/visual_{split}.npz');visual[split]={k:a[k][ix] for k in a.files}
    rows=[];radii={};cols=conditional_columns(frames['train'],True);train=frames['train'];val=frames['validation'];weight=row_weights(train)*len(train)
    variants={'numeric':[],'diagnostics':['diagnostics'],'appearance':['appearance'],'visual_motion':['diagnostics','motion'],'all_visual':['diagnostics','appearance','motion']}
    for variant,keys in variants.items():
        x={s:np.concatenate([f[cols].to_numpy(),*[visual[s][k] for k in keys]],axis=1) for s,f in frames.items()}
        for leaves in [7,15]:
            m=LGBMRegressor(objective='quantile',alpha=.9,n_estimators=180,num_leaves=leaves,min_child_samples=150,learning_rate=.04,reg_lambda=10,n_jobs=2,verbosity=-1,random_state=20261120)
            m.fit(x['train'],np.log1p(train.gate_normalized_error),sample_weight=weight)
            radius=np.maximum(np.expm1(m.predict(x['validation'])),1e-5);name=f'{variant}_leaves{leaves}'
            rows+=radius_metrics(val,radius,.9,name);radii[name]=radius;joblib.dump(m,out/(name+'.joblib'))
            pd.DataFrame(rows).to_csv(out/'validation_metrics.csv',index=False)
            print(json.dumps(dict(variant=variant,leaves=leaves,elapsed_s=time.perf_counter()-started)),flush=True)
    np.savez_compressed(out/'validation_radii.npz',**radii)
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,point_model='numeric frozen gate; training environment-OOF errors',columns=cols,variants=variants,elapsed_s=time.perf_counter()-started,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
