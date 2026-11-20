#!/usr/bin/env python3
"""Shared-coefficient multiscale velocity forecast: a causal learned linear rival.

AI-assisted: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
The linear predictor is translation/rotation equivariant, with no source identities.
"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset,balanced_weights
from shuwei_c.q1 import aggregate_state

LAGS=[1,3,6,12]


def features(record):
    t,p,bad=record.hands();c=p.mean(axis=2);origin=np.arange(2,len(t)-1)
    vectors=[]
    for lag in LAGS:
        previous=np.maximum(origin-lag,0)
        vectors.append((c[origin]-c[previous])/(t[origin]-t[previous])[:,None,None])
    return t,c,bad,origin,np.stack(vectors,axis=-1)


def main():
    data=Dataset();out=Path('artifacts/c_q1_v1')
    noise=json.loads((out/'state_model.json').read_text())['base_training_noise']
    train=data.records('train');validation=data.records('validation')
    model={};rows=[];selection=[];uncertainty=[];rng=np.random.default_rng(20261120)
    for horizon in ['one_step','0.1s']:
        grams=[];targets=[];metadata=[]
        for rec in train:
            t,c,bad,origin,x=features(rec)
            target=origin+1 if horizon=='one_step' else np.searchsorted(t,t[origin]+.1-1e-9)
            take=target<len(t);origin=origin[take];target=target[take];x=x[take]
            y=(c[target]-c[origin])/(t[target]-t[origin])[:,None,None]
            valid=~bad[target]&~bad[origin]
            for lag in LAGS:valid&=~bad[np.maximum(origin-lag,0)]
            xx=x[valid].reshape(-1,len(LAGS))/noise[rec.dataset]['hand_scale']
            yy=y[valid].ravel()/noise[rec.dataset]['hand_scale']
            if not len(yy):continue
            grams.append(xx.T@xx/len(yy));targets.append(xx.T@yy/len(yy))
            metadata.append((rec.environment,rec.episode))
        w=balanced_weights([r[0] for r in metadata],[r[1] for r in metadata])
        gram=np.einsum('n,nij->ij',w,np.array(grams));target=np.einsum('n,ni->i',w,np.array(targets))
        prior=np.array([1.,0.,0.,0.]);all_candidates=[]
        for penalty in [.001,.01,.1,1.,10.,100.]:
            coef=np.linalg.solve(gram+penalty*np.eye(4),target+penalty*prior)
            trial=[]
            for rec in validation:
                t,c,bad,origin,x=features(rec)
                end=origin+1 if horizon=='one_step' else np.searchsorted(t,t[origin]+.1-1e-9)
                take=end<len(t);origin=origin[take];end=end[take];x=x[take]
                pred=c[origin]+np.einsum('nhck,k->nhc',x,coef)*(t[end]-t[origin])[:,None,None]
                for j,hand in enumerate(['left','right']):
                    for context in ['all','recent_flag','unflagged_history']:
                        history=bad[origin,j]|bad[origin-1,j]
                        valid=~bad[end,j]&(history if context=='recent_flag' else ~history if context=='unflagged_history' else True)
                        if not valid.any():continue
                        error=np.linalg.norm(pred[valid,j]-c[end[valid],j],axis=-1)
                        trial.append(dict(episode=rec.episode,environment=rec.environment,dataset=rec.dataset,split=rec.split,
                            method='multiscale_linear',horizon=horizon,hand=hand,context=context,n=len(error),
                            mean_euclidean_error=float(error.mean()),p90_error=float(np.quantile(error,.9)),
                            actual_horizon_mean_s=float(np.mean(t[end[valid]]-t[origin[valid]]))))
            f=pd.DataFrame(trial);g=f[f.context=='all'].copy()
            g['scaled']=[v/noise[ds]['hand_scale'] for ds,v in zip(g.dataset,g.mean_euclidean_error)]
            score=float(g.groupby(['environment','episode']).scaled.mean().groupby('environment').mean().mean())
            selection.append(dict(horizon=horizon,penalty=penalty,normalized_error=score))
            all_candidates.append((score,coef,penalty,trial))
        score,coef,penalty,trial=min(all_candidates,key=lambda r:r[0]);rows.extend(trial)
        model[horizon]=dict(coefficients=coef.tolist(),penalty=penalty,lag_frames=LAGS,normalized_validation_error=score)
        # Environment bootstrap of fitted coefficients with chosen penalty held fixed.
        envs=sorted(set(r[0] for r in metadata));env_grams=[];env_targets=[]
        for env in envs:
            ix=[i for i,r in enumerate(metadata) if r[0]==env]
            env_grams.append(np.mean(np.array(grams)[ix],axis=0));env_targets.append(np.mean(np.array(targets)[ix],axis=0))
        estimates=[]
        for _ in range(2000):
            ix=rng.integers(0,len(envs),len(envs))
            a=np.mean(np.array(env_grams)[ix],axis=0);b=np.mean(np.array(env_targets)[ix],axis=0)
            estimates.append(np.linalg.solve(a+penalty*np.eye(4),b+penalty*prior))
        for j,lag in enumerate(LAGS):
            uncertainty.append(dict(horizon=horizon,lag_frames=lag,coefficient=coef[j],
                low=float(np.quantile(np.array(estimates)[:,j],.025)),high=float(np.quantile(np.array(estimates)[:,j],.975))))
    pd.DataFrame(selection).to_csv(out/'multiscale_parameter_sensitivity.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(out/'multiscale_coefficient_uncertainty.csv',index=False)
    (out/'multiscale_model.json').write_text(json.dumps(model,indent=2)+'\n')
    current=pd.read_csv(out/'state_validation_by_record.csv');current=current[current.method!='multiscale_linear']
    combined=pd.concat([current,pd.DataFrame(rows)],ignore_index=True)
    combined.to_csv(out/'state_validation_by_record.csv',index=False)
    aggregate_state(combined).to_csv(out/'state_validation_summary.csv',index=False)
    print(json.dumps(model),flush=True)


if __name__=='__main__':main()
