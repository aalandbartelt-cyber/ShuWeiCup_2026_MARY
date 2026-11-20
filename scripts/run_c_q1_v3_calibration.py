#!/usr/bin/env python3
"""Conditional quantile radii and causal residual adaptation on the numeric gate.

AI-assisted by Codex, 2026-11-20. Empirical group calibration; no exchangeability
or arbitrary missing-observation guarantee is claimed.
"""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
from shuwei_c.q1v3_state import row_weights
from shuwei_c.q1v2_state import weighted_quantile
from shuwei_c.q1v3_calibration import mature_residual_features,conditional_columns,calibration_shift,radius_metrics


def model(level,leaves):
    return LGBMRegressor(objective='quantile',alpha=level,n_estimators=180,num_leaves=leaves,max_depth=-1,
        min_child_samples=150,learning_rate=.04,reg_lambda=10.,n_jobs=2,verbosity=-1,random_state=20261120)


def fit(frame,columns,level,leaves):
    m=model(level,leaves);m.fit(frame[columns],np.log1p(frame.gate_normalized_error),sample_weight=row_weights(frame)*len(frame));return m


def main():
    out=Path('artifacts/c_q1_v3/calibration_screen');out.mkdir(exist_ok=True);source=Path('artifacts/c_q1_v3/numeric_oof');started=time.perf_counter()
    train=mature_residual_features(pd.read_parquet(source/'train_oof.parquet'));val=mature_residual_features(pd.read_parquet(source/'validation.parquet'))
    train.to_parquet(out/'train_features.parquet',index=False);val.to_parquet(out/'validation_features.parquet',index=False)
    levels=[.5,.8,.9,.95];rows=[];radii={};parameters={}
    for level in levels:
        constant=weighted_quantile(train.gate_normalized_error,row_weights(train),level);radius=np.full(len(val),constant)
        rows+=radius_metrics(val,radius,level,'global_quantile');radii[f'global_quantile_{level}']=radius
        for online in [False,True]:
            cols=conditional_columns(train,online)
            for leaves in [7,15]:
                name=f'quantile_{"online" if online else "static"}_leaves{leaves}';oof=np.zeros(len(train))
                for fi,hi in GroupKFold(4).split(train,groups=train.environment):
                    f=train.iloc[fi];held=train.iloc[hi];m=fit(f,cols,level,leaves)
                    oof[hi]=np.maximum(np.expm1(m.predict(held[cols])),1e-5)
                correction=calibration_shift(train,oof,level);m=fit(train,cols,level,leaves)
                base=np.maximum(np.expm1(m.predict(val[cols])),1e-5);joblib.dump(m,out/f'{name}_{level}.joblib')
                parameters[f'{name}_{level}']=dict(columns=cols,correction=correction)
                for scheme in ['raw','global','dataset']:
                    shift=0. if scheme=='raw' else (correction['global_shift'] if scheme=='global' else np.array([correction['dataset_shift'][ds] for ds in val.dataset]))
                    radius=np.maximum(base+shift,1e-5);method=name+'_'+scheme
                    rows+=radius_metrics(val,radius,level,method);radii[f'{method}_{level}']=radius
                print(json.dumps(dict(level=level,method=name,correction=correction)),flush=True)
        pd.DataFrame(rows).to_csv(out/'validation_metrics.csv',index=False)
    np.savez_compressed(out/'validation_radii.npz',**radii);(out/'parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
    manifest=dict(elapsed_s=time.perf_counter()-started,test_used=False,
        interpretation='empirical conditional quantiles from environment OOF point residuals; mature feedback uses only returned qualified references; calibration crossfit does not establish a distribution-free conditional guarantee',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_calibration.py')]})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
