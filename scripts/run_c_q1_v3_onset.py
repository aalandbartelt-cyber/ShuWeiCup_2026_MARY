#!/usr/bin/env python3
"""Direct onset-within-window risk, distinct from endpoint bad-flag probability.

Current state must be good. Whole-environment OOF calibrates alert thresholds;
same-capacity numeric/video ablations use no future input or visual PCA.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score,roc_auc_score
from lightgbm import LGBMClassifier
from shuwei_c import q1v2_quality as q
from shuwei_c.q1v3_shared_risk import features
from shuwei_c.data import Dataset
from run_c_q1_v3_articulated_risk import select_visual


def baseline(f):
    w=q.weights(f);y=f.onset_within_horizon.to_numpy();global_rate=float(w@y);p={}
    for ds,g in f.groupby('dataset'):
        use=(f.dataset==ds).to_numpy();p[ds]=float((w[use]@y[use]+global_rate/len(f))/(w[use].sum()+1/len(f)))
    return p


def predict_base(parameters,f):return np.array([parameters[d] for d in f.dataset])


def metrics(f,p,name,split):
    rows=[]
    for ds in ['ALL',*f.dataset.unique()]:
        take=np.ones(len(f),bool) if ds=='ALL' else (f.dataset==ds).to_numpy();g=f.loc[take];y=g.onset_within_horizon.to_numpy().astype(int);prob=p[take];w=q.weights(g);clip=np.clip(prob,1e-7,1-1e-7)
        rows.append(dict(split=split,method=name,dataset=ds,n=len(g),positive=int(y.sum()),brier=float(w@((prob-y)**2)),
            log_loss=float(w@(-y*np.log(clip)-(1-y)*np.log1p(-clip))),ap=float(average_precision_score(y,prob,sample_weight=w)) if y.any() else None,
            auc=float(roc_auc_score(y,prob,sample_weight=w)) if len(np.unique(y))==2 else None))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--horizon',type=float,choices=[.1,.5],default=.5);args=parser.parse_args()
    out=Path(f'artifacts/c_q1_v3/onset_h{args.horizon}');out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Completed experiment exists')
    started=time.perf_counter();frames={};fullframes={};raw={};cols=None
    for split in ['train','validation']:
        f=pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_{split}_{args.horizon}.parquet');fullframes[split]=f;mask=f.current_state.to_numpy()==0;frames[split]=f.loc[mask].reset_index(drop=True)
        source=pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_{split}_0.1.parquet');ix=pd.MultiIndex.from_frame(source[['episode','origin']]).get_indexer(pd.MultiIndex.from_frame(f[['episode','origin']]))
        assert (ix>=0).all();np.testing.assert_array_equal(source.time_s.to_numpy()[ix],f.time_s.to_numpy());cols=q.columns_for(f,'geometry');numeric=f[cols].to_numpy()
        diagnostics=np.load(f'artifacts/c_q1_v3/video_quality/visual_{split}.npz')['diagnostics'][ix];video,_,_=features(f,diagnostics,True)
        articulated=np.load(f'artifacts/c_q1_v3/articulated_features/quality_{split}.npz')['features'][ix]
        raw[split]={'numeric':numeric[mask],'diagnostics':np.c_[numeric,video[:,:,9:].reshape(len(f),-1)][mask],
                    'articulated':np.c_[numeric,select_visual(articulated,'scalar').reshape(len(f),-1)][mask]}
    train=frames['train'];val=frames['validation'];folds=list(GroupKFold(4).split(train,groups=train.environment));oof={};vp={};rows=[];events=[];train_prior=np.zeros(len(train))
    for fi,hi in folds:train_prior[hi]=predict_base(baseline(train.iloc[fi]),train.iloc[hi])
    oof['frequency']=train_prior;vp['frequency']=predict_base(baseline(train),val);joblib.dump(baseline(train),out/'frequency.joblib')
    for kind in raw['train']:
        for leaves in [7,15]:
            name=f'{kind}_leaves{leaves}';pred=np.zeros(len(train))
            def fit(f,x):
                m=LGBMClassifier(objective='binary',n_estimators=150,num_leaves=leaves,min_child_samples=200,learning_rate=.03,reg_lambda=20.,n_jobs=2,verbosity=-1,random_state=20261120)
                m.fit(x,f.onset_within_horizon.astype(int),sample_weight=q.weights(f)*len(f));return m
            for fold,(fi,hi) in enumerate(folds):
                m=fit(train.iloc[fi],raw['train'][kind][fi]);pred[hi]=m.predict_proba(raw['train'][kind][hi])[:,1];joblib.dump(m,out/f'fold{fold}_{name}.joblib')
            m=fit(train,raw['train'][kind]);joblib.dump(m,out/f'full_{name}.joblib');oof[name]=pred;vp[name]=m.predict_proba(raw['validation'][kind])[:,1]
            print(json.dumps(dict(method=name,elapsed_s=time.perf_counter()-started)),flush=True)
    records=Dataset().records('validation');vf=fullframes['validation'];mask=vf.current_state.to_numpy()==0
    for name in oof:
        rows+=metrics(train,oof[name],name,'train_oof');rows+=metrics(val,vp[name],name,'validation')
        # q.event_metrics consumes only the any-event score in onset mode.
        # This adapter is NOT a four-state endpoint probability model.
        tp=np.c_[1-oof[name],oof[name],np.zeros((len(train),2))]
        ep=np.zeros(len(vf));ep[mask]=vp[name];adapter=np.c_[1-ep,ep,np.zeros((len(vf),2))]
        for level in [.01,.05,.1]:
            threshold=q.select_threshold(tp,train,level,'onset');values,_=q.event_metrics(records,vf,adapter,threshold,args.horizon,'onset')
            events.append(dict(method=name,nominal_false_alarm=level,threshold=threshold,**values))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(events).to_csv(out/'events.csv',index=False)
    for split,f,p in [('train_oof',train,oof),('validation',val,vp)]:
        saved=f[['episode','environment','dataset','origin','target','time_s','target_time_s','current_state','onset_within_horizon']].copy()
        for name,prob in p.items():saved[name+'_onset_probability']=prob
        saved.to_parquet(out/f'{split}.parquet',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,horizon=args.horizon,elapsed_s=time.perf_counter()-started,numeric_columns=cols,
        target='At least one new supplied bad-flag onset within the future horizon, conditional on both hands currently good; distinct from endpoint bad probability',
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),folds=[dict(fold=i,held_environments=sorted(train.iloc[hi].environment.unique())) for i,(_,hi) in enumerate(folds)]),indent=2)+'\n')


if __name__=='__main__':main()
