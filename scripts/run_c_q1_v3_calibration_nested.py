#!/usr/bin/env python3
"""Outer-environment calibration audit with fully nested point residual fitting.

Every outer held environment is excluded from all point/quantile fitting and
calibration shifts. Held-record feedback is permitted only after reference arrival.
"""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from shuwei_c.data import Dataset
from shuwei_c.motion import estimate_noise
from shuwei_c.q1v3_state import pack_records,fit_distance
from shuwei_c.q1v3_learning import fit_gate,predict_gate
from shuwei_c.q1v3_calibration import mature_residual_features,conditional_columns,calibration_shift,radius_metrics
from shuwei_c.q1v3_feedback import adapt_radii
from run_c_q1_v3_numeric_oof import attach
from run_c_q1_v3_calibration import fit


def main():
    out=Path('artifacts/c_q1_v3/calibration_nested');out.mkdir(exist_ok=True);start=time.perf_counter()
    data=Dataset();records=data.records('train');config=json.loads(Path('configs/c_q1_v2.json').read_text())
    outer=json.loads(Path('artifacts/c_q1_v3/numeric_oof/folds.json').read_text());allheld=pd.read_parquet('artifacts/c_q1_v3/numeric_oof/train_oof.parquet')
    levels=[.5,.8,.9,.95];rates=[0.,.005,.01,.02,.05,.1];frames=[];radii={str(rate):[] for rate in rates};logs=[]
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    for spec in outer:
        fold=spec['fold'];fitting=[r for r in records if r.environment in spec['training_environments']];cache=out/f'fold{fold}_inner_residuals.parquet'
        if cache.exists():train=pd.read_parquet(cache)
        else:
            parts=[]
            for inner,(fi,hi) in enumerate(GroupKFold(3).split(fitting,groups=[r.environment for r in fitting])):
                fr=[fitting[i] for i in fi];hr=[fitting[i] for i in hi];noise=estimate_noise(fr)
                a=pack_records(fr,.1,config,noise);b=pack_records(hr,.1,config,noise);base=fit_distance(a,.01,True);pred=[]
                for seed in [20261120,20261121,20261122]:
                    m=fit_gate(a,base,10,.001,seed);torch.save(m,out/f'fold{fold}_inner{inner}_s{seed}.pt');pred.append(predict_gate(b,m))
                parts.append(attach(b,dict(gate=np.mean(pred,axis=0)),inner));print(json.dumps(dict(fold=fold,inner=inner,elapsed_s=time.perf_counter()-start)),flush=True)
            train=mature_residual_features(pd.concat(parts,ignore_index=True));train.to_parquet(cache,index=False)
        held=mature_residual_features(allheld[allheld.fold==fold].reset_index(drop=True));cols=conditional_columns(train,True);base=[];corrections={}
        assert not set(train.environment)&set(held.environment)
        for level in levels:
            cp=np.zeros(len(train))
            for fi,hi in GroupKFold(3).split(train,groups=train.environment):
                m=fit(train.iloc[fi],cols,level,15);cp[hi]=np.maximum(np.expm1(m.predict(train.iloc[hi][cols])),1e-5)
            correction=calibration_shift(train,cp,level);corrections[str(level)]=correction
            m=fit(train,cols,level,15);raw=np.maximum(np.expm1(m.predict(held[cols])),1e-5)
            shift=np.array([correction['dataset_shift'].get(ds,correction['global_shift']) for ds in held.dataset]);base.append(np.maximum(raw+shift,1e-5))
        base=np.stack(base,axis=1);frames.append(held)
        for rate in rates:radii[str(rate)].append(adapt_radii(held,base,levels,rate)[0])
        logs.append(dict(fold=fold,held_environments=sorted(held.environment.unique()),corrections=corrections));print(json.dumps(dict(fold=fold,complete=True,elapsed_s=time.perf_counter()-start)),flush=True)
    frame=pd.concat(frames,ignore_index=True);rows=[];groups=[];full={key:np.concatenate(value) for key,value in radii.items()}
    for rate in rates:
        r=full[str(rate)]
        for k,level in enumerate(levels):
            result=radius_metrics(frame,r[:,k],level,f'feedback_{rate}')
            for item in result:item['rate']=rate
            rows+=result
        for env,g in frame.groupby('environment'):
            result=radius_metrics(g.reset_index(drop=True),r[g.index,2],.9,f'feedback_{rate}')
            for item in result:item.update(rate=rate,environment=env)
            groups+=result
    frame.to_parquet(out/'train_outer.parquet',index=False);np.savez_compressed(out/'radii.npz',**full)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(groups).to_csv(out/'environment_metrics.csv',index=False)
    (out/'folds.json').write_text(json.dumps(logs,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(dict(test_used=False,elapsed_s=time.perf_counter()-start,levels=levels,rates=rates,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_feedback.py')]},interpretation='Empirical outer-environment audit; all inner point, quantile and shift fitting excludes the outer environment. No distribution-free coverage guarantee.'),indent=2)+'\n')


if __name__=='__main__':main()
