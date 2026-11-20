"""Causal rotation-equivariant expert displacements and direct-distance fitting.

AI-assisted by Codex, 2026-11-20. All feature samples precede the forecast origin.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .data import balanced_weights
from .q1v2_state import matched_targets,history_features,nearest_indices
from .q1v2_quality import status_history

LAGS=[1/30,2/30,.1,.15,.2,.3,.4]
DATASETS=['EgoDemo','EgoPro','EgoStandard']


def polynomial_expert(t,c,origins,delta,window,degree=2):
    offsets=np.arange(window-1,-1,-1);ix=np.maximum(origins[:,None]-offsets[None,:],0)
    age=(t[ix]-t[origins,None])/.1
    design=np.stack([age**k for k in range(degree+1)],axis=-1)
    coef=np.einsum('nkw,nwhc->nkhc',np.linalg.pinv(design),c[ix]-c[origins,None,:,:])
    query=np.stack([(delta/.1)**k for k in range(degree+1)],axis=-1)
    return np.einsum('nk,nkhc->nhc',query,coef)


def pack_records(records,horizon,config,noise):
    frames=[];expert_list=[];feature_list=[];target_list=[];feature_names=None
    for rec in records:
        t,p,bad=rec.hands();c=p.mean(axis=2);scale=noise[rec.dataset]['hand_scale']
        i,j=matched_targets(t,horizon,config['warmup_s'],config['target_tolerance_steps']);delta=t[j]-t[i]
        v,speed,turn,history_bad=history_features(t,c,bad,i,LAGS)
        experts=[v[...,k]*delta[:,None,None] for k in range(len(LAGS))]
        for window in [5,9,13]:experts.append(polynomial_expert(t,c,i,delta,window))
        e=np.stack(experts,axis=-1)/scale;norm=np.linalg.norm(e,axis=2);features={}
        for k in range(e.shape[-1]):features[f'log_expert_norm_{k}']=np.log1p(norm[...,k])
        for a in range(7):
            for b in range(a+1,7):features[f'velocity_cos_{a}_{b}']=np.sum(e[...,a]*e[...,b],axis=2)/(norm[...,a]*norm[...,b]+1e-8)
        features['flag_now']=bad[i].astype(float);features['flag_sampled_history']=history_bad.astype(float)
        features['log_status_age']=np.log1p(status_history(t,bad)[i]);features['log_horizon']=np.full((len(i),2),np.log1p(horizon))
        shape=p[i]-c[i,:,None,:];singular=np.linalg.svd(shape/scale,compute_uv=False)
        for k in range(3):features[f'shape_singular_{k}']=singular[...,k]
        relative=(c[i,1]-c[i,0])/scale;features['log_interhand_distance']=np.repeat(np.log1p(np.linalg.norm(relative,axis=-1))[:,None],2,axis=1)
        head=rec.streams['head'];hi=np.searchsorted(head['time'],t[i],side='right')-1;hp=head['position'][hi].reshape(-1,3)
        features['log_head_distance']=np.log1p(np.linalg.norm(c[i]-hp[:,None,:],axis=-1)/scale)
        features['side']=np.tile([0.,1.],(len(i),1))
        for dataset in DATASETS:features['dataset_'+dataset]=np.full((len(i),2),float(dataset==rec.dataset))
        # Mature residual history: a previous linear forecast has a known target
        # by now. These target values are current/past observations, never future.
        lag=np.maximum(np.minimum(nearest_indices(t,t[i]-horizon),i-1),1)
        previous=(c[lag]-c[lag-1])/np.maximum(t[lag]-t[lag-1],1e-9)[:,None,None]
        matured_error=np.linalg.norm(c[lag]+previous*(t[i]-t[lag])[:,None,None]-c[i],axis=-1)/scale
        matured_good=~bad[lag]&~bad[i]&~bad[lag-1]
        features['matured_linear_error']=np.where(matured_good,np.log1p(matured_error),0.)
        features['matured_error_known']=matured_good.astype(float)
        x=np.stack(list(features.values()),axis=-1);feature_names=list(features)
        target=(c[j]-c[i])/scale
        valid=~bad[j]&np.isfinite(target).all(axis=-1)&np.isfinite(x).all(axis=-1)&np.isfinite(e).all(axis=(2,3))
        for hand in range(2):
            m=valid[:,hand];k=np.flatnonzero(m)
            frames.append(pd.DataFrame(dict(episode=rec.episode,environment=rec.environment,dataset=rec.dataset,split=rec.split,
                origin=i[k],target=j[k],origin_s=t[i[k]],target_s=t[j[k]],hand=hand,horizon=horizon,scale=scale,
                recent_bad=history_bad[k,hand])))
            expert_list.append(e[m,hand]);feature_list.append(x[m,hand]);target_list.append(target[m,hand])
    return dict(meta=pd.concat(frames,ignore_index=True),experts=np.concatenate(expert_list).astype(np.float32),
        features=np.concatenate(feature_list).astype(np.float32),target=np.concatenate(target_list).astype(np.float32),feature_names=feature_names)


def row_weights(frame):
    # Explicit equal hand, recording and environment weights.
    keys=frame[['environment','episode','hand']].drop_duplicates();keys['group_weight']=balanced_weights(keys.environment,keys.episode)
    counts=frame.groupby(['environment','episode','hand']).size().rename('count').reset_index();counts=counts.merge(keys,on=['environment','episode','hand'])
    counts['weight']=counts.group_weight/counts['count']
    return frame.merge(counts[['environment','episode','hand','weight']],on=['environment','episode','hand'],how='left',sort=False).weight.to_numpy()


def score(frame,error):
    return float(np.dot(row_weights(frame),error))


def fit_distance(pack,penalty=.001,dataset_specific=False):
    x=pack['experts'].astype(float);y=pack['target'].astype(float);w=row_weights(pack['meta']);k=x.shape[-1]
    code=np.array([DATASETS.index(ds) for ds in pack['meta'].dataset]);groups=3 if dataset_specific else 1
    prior=np.zeros((groups,k));prior[:,0]=1.
    def objective(flat):
        coef=flat.reshape(groups,k);a=coef[code] if dataset_specific else np.broadcast_to(coef[0],(len(x),k))
        error=np.einsum('nck,nk->nc',x,a)-y;norm=np.sqrt(np.sum(error**2,axis=1)+1e-8)
        grad_rows=np.einsum('nck,nc->nk',x,error)*(w/norm)[:,None]
        grad=np.stack([grad_rows[code==g].sum(axis=0) for g in range(3)]) if dataset_specific else grad_rows.sum(axis=0,keepdims=True)
        value=float(np.dot(w,norm))+penalty*np.mean((coef-prior)**2)
        grad+=2*penalty*(coef-prior)/coef.size
        if dataset_specific:
            delta=coef-coef.mean(axis=0);value+=penalty*np.mean(delta**2);grad+=2*penalty*delta/coef.size
        return value,grad.ravel()
    result=minimize(objective,prior.ravel(),jac=True,method='L-BFGS-B',bounds=[(-2,2)]*prior.size,
        options=dict(maxiter=180,ftol=1e-10,gtol=1e-7))
    return dict(coefficients=result.x.reshape(groups,k).tolist(),dataset_specific=dataset_specific,penalty=penalty,
        convergence=bool(result.success),iterations=result.nit,objective=float(result.fun))


def predict_distance(pack,spec):
    coef=np.array(spec['coefficients']);code=np.array([DATASETS.index(ds) for ds in pack['meta'].dataset])
    a=coef[code] if spec['dataset_specific'] else np.broadcast_to(coef[0],(len(code),coef.shape[-1]))
    return np.einsum('nck,nk->nc',pack['experts'],a)
