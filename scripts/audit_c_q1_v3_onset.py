#!/usr/bin/env python3
"""Video alignment falsification and paired environment audit of onset risk."""
from pathlib import Path
import json,sys,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from shuwei_c import q1v2_quality as q
from shuwei_c.q1v3_shared_risk import features
from shuwei_c.data import Dataset
from run_c_q1_v3_onset import metrics


def main():
    root=Path('artifacts/c_q1_v3/onset_h0.5');out=root/'audit';out.mkdir(exist_ok=True);source=json.loads((root/'manifest.json').read_text())
    f=pd.read_parquet('artifacts/c_q1_v2/quality_numerical/features_validation_0.5.parquet');old=pd.read_parquet('artifacts/c_q1_v2/quality_numerical/features_validation_0.1.parquet')
    ix=pd.MultiIndex.from_frame(old[['episode','origin']]).get_indexer(pd.MultiIndex.from_frame(f[['episode','origin']]))
    diag=np.load('artifacts/c_q1_v3/video_quality/visual_validation.npz')['diagnostics'][ix];v,_,_=features(f,diag,True);visual=v[:,:,9:].reshape(len(f),-1)
    numeric=f[source['numeric_columns']].to_numpy();mask=f.current_state.to_numpy()==0;frame=f[mask].reset_index(drop=True)
    model=joblib.load(root/'full_diagnostics_leaves7.joblib');train=pd.read_parquet(root/'train_oof.parquet');rows=[];events=[];predictions={};records=Dataset().records('validation');rng=np.random.default_rng(20261120)
    for control in ['aligned','lag_0.5s','lag_1s','lag_2s','permuted_within_episode']:
        z=visual.copy() if control=='aligned' else np.zeros_like(visual)
        if control!='aligned':
            for _,g in f.groupby('episode',sort=False):
                index=g.index.to_numpy()
                if control.startswith('lag'):
                    lag=float(control.split('_')[1][:-1]);take=np.searchsorted(g.time_s,g.time_s-lag,side='right')-1;known=take>=0;z[index[known]]=visual[index[take[known]]]
                else:z[index]=visual[rng.permutation(index)]
        p=model.predict_proba(np.c_[numeric,z][mask])[:,1];predictions[control]=p;rows+=metrics(frame,p,control,'validation')
        if control=='aligned':np.testing.assert_allclose(p,pd.read_parquet(root/'validation.parquet').diagnostics_leaves7_onset_probability,atol=1e-12)
        trainp=train.diagnostics_leaves7_onset_probability.to_numpy();tp=np.c_[1-trainp,trainp,np.zeros((len(train),2))]
        fullp=np.zeros(len(f));fullp[mask]=p;adapter=np.c_[1-fullp,fullp,np.zeros((len(f),2))]
        for level in [.01,.05,.1]:
            threshold=q.select_threshold(tp,train,level,'onset');result,_=q.event_metrics(records,f,adapter,threshold,.5,'onset');events.append(dict(control=control,nominal_false_alarm=level,threshold=threshold,**result))
    environments=[];bootstrap=[]
    for split in ['train_oof','validation']:
        g=pd.read_parquet(root/f'{split}.parquet');values=[]
        for env,group in g.groupby('environment'):
            y=group.onset_within_horizon.to_numpy();w=q.weights(group);n=group.numeric_leaves7_onset_probability.to_numpy();v=group.diagnostics_leaves7_onset_probability.to_numpy()
            ne=float(w@(n-y)**2);ve=float(w@(v-y)**2);values.append([ne,ve]);environments.append(dict(split=split,environment=env,n=len(group),positive=int(y.sum()),numeric_brier=ne,video_brier=ve,gain_pct=100*(1-ve/ne)))
        values=np.asarray(values);samples=rng.integers(len(values),size=(2000,len(values)));means=values[samples].mean(axis=1);gain=100*(1-means[:,1]/means[:,0]);bootstrap.append(dict(split=split,environment_count=len(values),gain_pct=100*(1-values[:,1].mean()/values[:,0].mean()),low_pct=float(np.quantile(gain,.025)),high_pct=float(np.quantile(gain,.975)),improved_environments=int((values[:,1]<values[:,0]).sum())))
    pd.DataFrame(rows).to_csv(out/'control_metrics.csv',index=False);pd.DataFrame(events).to_csv(out/'control_events.csv',index=False)
    pd.DataFrame(environments).to_csv(out/'environment_metrics.csv',index=False);pd.DataFrame(bootstrap).to_csv(out/'environment_bootstrap.csv',index=False);np.savez_compressed(out/'control_probabilities.npz',**predictions)
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,bootstrap_replicates=2000,scope='Fixed development-selected model, paired environment resampling; lag controls use earlier video, within-record permutation is an intentionally noncausal falsification control and is never deployed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
