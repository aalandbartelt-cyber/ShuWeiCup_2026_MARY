"""Inference-only v3 features from explicitly disclosed hand/head samples.

No target trajectory, recording identity lookup, video lookup or interpolation.
Sparse disclosed histories are accepted, but need separate empirical calibration.
"""
import numpy as np
import pandas as pd
from .q1v2_state import history_features,nearest_indices
from .q1v2_quality import status_history
from .q1v3_state import LAGS,DATASETS,polynomial_expert


def causal_pack(time,points,bad,head_time,head_position,dataset,scale,
                query_time,nominal_horizon=None):
    t=np.asarray(time,float);p=np.asarray(points,float);bad=np.asarray(bad,bool)
    ht=np.asarray(head_time,float);hp=np.asarray(head_position,float).reshape(-1,3)
    if len(t)<3 or p.shape!=(len(t),2,21,3) or bad.shape!=(len(t),2):raise ValueError('Need at least three disclosed two-hand observations with 21 points')
    if np.any(np.diff(t)<=0) or not np.isfinite(p).all():raise ValueError('Disclosed points must be finite and times strictly increasing')
    if query_time<t[-1]-1e-9 or scale<=0 or dataset not in DATASETS:raise ValueError('Invalid query, scale or dataset')
    if not len(ht) or len(ht)!=len(hp) or np.any(np.diff(ht)<=0):raise ValueError('Head times must be strictly increasing')
    hi=np.searchsorted(ht,t[-1],side='right')-1
    if hi<0 or not np.isfinite(hp[hi]).all():raise ValueError('No disclosed head observation at the forecast origin')
    c=p.mean(axis=2);i=np.array([len(t)-1]);delta=np.array([max(query_time-t[-1],0.)]);h=delta[0] if nominal_horizon is None else nominal_horizon
    if h<0:raise ValueError('Negative nominal horizon')
    velocity,_,_,history_bad=history_features(t,c,bad,i,LAGS)
    experts=[velocity[...,k]*delta[:,None,None] for k in range(len(LAGS))]
    for window in [5,9,13]:experts.append(polynomial_expert(t,c,i,delta,window))
    e=np.stack(experts,axis=-1)/scale;norm=np.linalg.norm(e,axis=2);features={}
    for k in range(e.shape[-1]):features[f'log_expert_norm_{k}']=np.log1p(norm[...,k])
    for a in range(7):
        for b in range(a+1,7):features[f'velocity_cos_{a}_{b}']=np.sum(e[...,a]*e[...,b],axis=2)/(norm[...,a]*norm[...,b]+1e-8)
    features['flag_now']=bad[i].astype(float);features['flag_sampled_history']=history_bad.astype(float)
    features['log_status_age']=np.log1p(status_history(t,bad)[i]);features['log_horizon']=np.full((1,2),np.log1p(h))
    singular=np.linalg.svd((p[i]-c[i,:,None,:])/scale,compute_uv=False)
    for k in range(3):features[f'shape_singular_{k}']=singular[...,k]
    features['log_interhand_distance']=np.repeat(np.log1p(np.linalg.norm((c[i,1]-c[i,0])/scale,axis=-1))[:,None],2,axis=1)
    features['log_head_distance']=np.log1p(np.linalg.norm(c[i]-hp[hi][None,None,:],axis=-1)/scale)
    features['side']=np.array([[0.,1.]])
    for ds in DATASETS:features['dataset_'+ds]=np.full((1,2),float(dataset==ds))
    lag=np.maximum(np.minimum(nearest_indices(t,t[i]-h),i-1),1)
    previous=(c[lag]-c[lag-1])/np.maximum(t[lag]-t[lag-1],1e-9)[:,None,None]
    error=np.linalg.norm(c[lag]+previous*(t[i]-t[lag])[:,None,None]-c[i],axis=-1)/scale
    known=~bad[lag]&~bad[i]&~bad[lag-1]
    features['matured_linear_error']=np.where(known,np.log1p(error),0.);features['matured_error_known']=known.astype(float)
    x=np.stack(list(features.values()),axis=-1)[0]
    return dict(meta=pd.DataFrame(dict(dataset=[dataset]*2,hand=[0,1],scale=[scale]*2)),
        experts=e[0].astype(np.float32),features=x.astype(np.float32),feature_names=list(features),
        origin_s=float(t[-1]),query_s=float(query_time),centres=c[-1].copy())
