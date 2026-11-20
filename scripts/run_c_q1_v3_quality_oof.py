#!/usr/bin/env python3
"""Nested-prior environment verification and visual falsification controls.

AI-assisted by Codex, 2026-11-20. Model choices are development-selected.
Outer held environments are excluded from every fitted prior/network.
"""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from shuwei_c import q1v2_quality as q
from shuwei_c import q1v3_quality as learn
from shuwei_c.data import Dataset


def inner_prior(frame):
    p=np.zeros((len(frame),4))
    for fi,hi in GroupKFold(3).split(frame,groups=frame.environment):p[hi]=q.predict_baseline(q.fit_baseline(frame.iloc[fi]),frame.iloc[hi])
    return p


def save_predictions(frame,probs,path):
    f=frame[['episode','environment','dataset','origin','target','time_s','target_time_s','current_state','target_state']].copy()
    for method,p in probs.items():
        for k in range(4):f[f'{method}_p{k}']=p[:,k]
    f.to_parquet(path,index=False)


def main():
    out=Path('artifacts/c_q1_v3/quality_oof');out.mkdir(exist_ok=True);source=Path('artifacts/c_q1_v3/video_quality');started=time.perf_counter()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
    train=pd.read_parquet('artifacts/c_q1_v2/quality_numerical/features_train_0.1.parquet');val=pd.read_parquet('artifacts/c_q1_v2/quality_numerical/features_validation_0.1.parquet')
    cols=q.columns_for(train,'geometry');numeric=train[cols].to_numpy();diagnostics=np.load(source/'visual_train.npz')['diagnostics'];seeds=[20261120,20261121,20261122]
    oof={name:np.zeros((len(train),4)) for name in ['transition','numeric','diagnostics','mixture']};folds=[]
    for fold,(fi,hi) in enumerate(GroupKFold(4).split(train,groups=train.environment)):
        fitting=train.iloc[fi];held=train.iloc[hi];prior=inner_prior(fitting);base=q.fit_baseline(fitting);hp=q.predict_baseline(base,held);oof['transition'][hi]=hp
        for variant,features in [('numeric',numeric),('diagnostics',np.c_[numeric,diagnostics])]:
            values=[]
            for seed in seeds:
                model=learn.fit(fitting,features[fi],prior,seed);torch.save(dict(model,baseline=base,cols=cols,variant=variant),out/f'fold{fold}_{variant}_seed{seed}.pt')
                values.append(learn.predict(features[hi],hp,model))
            oof[variant][hi]=np.mean(values,axis=0)
        oof['mixture'][hi]=.5*(oof['numeric'][hi]+oof['diagnostics'][hi])
        scores={name:next(r['brier'] for r in q.metrics(held,p[hi],name) if r['dataset']=='ALL' and r['condition']=='all' and r['target']=='any') for name,p in oof.items()}
        folds.append(dict(fold=fold,held_environments=sorted(held.environment.unique()),scores=scores));print(json.dumps(folds[-1]),flush=True)
    save_predictions(train,oof,out/'train_oof.parquet');rows=[]
    for name,p in oof.items():
        values=q.metrics(train,p,name)
        for r in values:r['split']='train_oof'
        rows+=values
    # Reuse the actual full-training checkpoints from the screen.
    cache=np.load(source/'probabilities.npz');vp={}
    vp['transition']=cache['transition']
    for variant in ['numeric','diagnostics']:vp[variant]=np.mean([cache[f'{variant}_r0.001_s{s}_e10'] for s in seeds],axis=0)
    vp['mixture']=.5*(vp['numeric']+vp['diagnostics'])
    visual_val=np.load(source/'visual_validation.npz')['diagnostics'];rng=np.random.default_rng(20261120)
    for control in ['lag_1s','lag_2s','permuted_within_episode']:
        corrupted=np.zeros_like(visual_val)
        for _,g in val.groupby('episode',sort=False):
            ix=g.index.to_numpy()
            if control.startswith('lag'):
                lag=float(control.split('_')[1][:-1]);take=np.searchsorted(g.time_s,g.time_s-lag,side='right')-1;known=take>=0
                corrupted[ix[known]]=visual_val[ix[take[known]]]
            else:corrupted[ix]=visual_val[rng.permutation(ix)]
        values=[]
        for seed in seeds:
            model=torch.load(source/f'diagnostics_r0.001_s{seed}_e10.pt',map_location='cpu',weights_only=False)
            values.append(learn.predict(np.c_[val[cols].to_numpy(),corrupted],vp['transition'],model))
        vp['mixture_'+control]=.5*(vp['numeric']+np.mean(values,axis=0))
    for name,p in vp.items():
        values=q.metrics(val,p,name)
        for r in values:r['split']='validation'
        rows+=values
    save_predictions(val,vp,out/'validation.parquet');pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    events=[];data=Dataset();records=data.records('validation')
    for name in oof:
        for mode in ['onset','recovery']:
            for level in [.01,.05,.1]:
                threshold=q.select_threshold(oof[name],train,level,mode);value,_=q.event_metrics(records,val,vp[name],threshold,.1,mode)
                events.append(dict(method=name,mode=mode,nominal_false_alarm=level,threshold=threshold,**value))
    pd.DataFrame(events).to_csv(out/'events.csv',index=False);(out/'folds.json').write_text(json.dumps(folds,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,elapsed_s=time.perf_counter()-started,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_quality.py')]},
        limitations='Hyperparameters and equal numeric/visual mixture examined on development validation; within-episode permutation is a deliberately noncausal falsification control, never deployed. Lag controls use only earlier visual diagnostics.'),indent=2)+'\n')


if __name__=='__main__':main()
