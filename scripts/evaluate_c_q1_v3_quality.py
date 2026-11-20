#!/usr/bin/env python3
"""Fixed quality/onset evaluation on the previously observed test recordings."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
import torch
from shuwei_c.data import Dataset
from shuwei_c import q1v2_quality as q
from shuwei_c.q1v3_shared_risk import features
from shuwei_c.q1v3_head_diagnostics import joint
from shuwei_c.q1v3_articulated_risk import predict
from run_c_q1_v3_onset import metrics,predict_base


def main():
    root=Path('artifacts/c_q1_v3/frozen_quality');spec=json.loads((root/'freeze.json').read_text());out=root/'test';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Fixed quality test evaluation already exists')
    for p,h in {**spec['inherited_v2_hashes'],**spec['source_hashes']}.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    visual=root/'test_video/diagnostics';assert len(json.loads((visual/'manifest.json').read_text()))==14
    started=datetime.now(timezone.utc).isoformat();torch.set_num_threads(2);records=Dataset().records('test');config=json.loads(Path('configs/c_q1_v2.json').read_text());endpoint_rows=[];events=[];onset_rows=[];frames={}
    for h,entry in spec['endpoint'].items():
        f=q.dataset_frame(records,float(h),config);frames[h]=f;n,flag,_=features(f,None,False);raw=np.concatenate([n,flag[...,None]],axis=-1)
        prior=q.predict_baseline(joblib.load(entry['prior_path']),f);p=np.mean([predict(raw,prior,torch.load(path,map_location='cpu',weights_only=False)) for path in entry['checkpoints']],axis=0)
        for name,prob in [('baseline',prior),('numeric_shared',p)]:
            values=q.metrics(f,prob,name)
            for r in values:r['horizon']=float(h)
            endpoint_rows+=values
        saved=f[['episode','environment','dataset','origin','target','time_s','target_time_s','current_state','target_state']].copy()
        for k in range(4):saved[f'p{k}']=p[:,k]
        saved.to_parquet(out/f'endpoint_{h}.parquet',index=False)
        for mode,thresholds in entry['thresholds'].items():
            for level,threshold in thresholds.items():
                result,_=q.event_metrics(records,f,p,threshold,float(h),mode);events.append(dict(target='endpoint_probability',method='numeric_shared',horizon=float(h),mode=mode,nominal_false_alarm=float(level),threshold=threshold,**result))
    entry=spec['onset'];f=frames[str(entry['horizon'])];mask=f.current_state.to_numpy()==0;filtered=f.loc[mask].reset_index(drop=True);diagnostics=joint(f,visual);v,_,_=features(f,diagnostics,True)
    numerical=f[entry['numeric_columns']].to_numpy();raw={'numeric_leaves7':numerical[mask],'diagnostics_leaves7':np.c_[numerical,v[:,:,9:].reshape(len(f),-1)][mask]}
    saved=filtered[['episode','environment','dataset','origin','target','time_s','target_time_s','current_state','onset_within_horizon']].copy()
    for name,model in entry['models'].items():
        fitted=joblib.load(model['path']);p=predict_base(fitted,filtered) if name=='frequency' else fitted.predict_proba(raw[name])[:,1]
        onset_rows+=metrics(filtered,p,name,'fixed_test');saved[name+'_onset_probability']=p
        full=np.zeros(len(f));full[mask]=p;adapter=np.c_[1-full,full,np.zeros((len(f),2))]
        for level,threshold in model['thresholds'].items():
            result,_=q.event_metrics(records,f,adapter,threshold,entry['horizon'],'onset');events.append(dict(target='onset_within_window',method=name,horizon=entry['horizon'],mode='onset',nominal_false_alarm=float(level),threshold=threshold,**result))
    saved.to_parquet(out/'onset.parquet',index=False);pd.DataFrame(endpoint_rows).to_csv(out/'endpoint_metrics.csv',index=False);pd.DataFrame(onset_rows).to_csv(out/'onset_metrics.csv',index=False);pd.DataFrame(events).to_csv(out/'events.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),freeze_timestamp=spec['timestamp_utc'],freeze_sha256=hashlib.sha256((root/'freeze.json').read_bytes()).hexdigest(),records=len(records),environments=len({r.environment for r in records}),interpretation='Fixed-version reevaluation of previously observed official test; all model/threshold/preprocessing choices fixed before test video extraction',fitted_from_test=False),indent=2)+'\n')


if __name__=='__main__':main()
