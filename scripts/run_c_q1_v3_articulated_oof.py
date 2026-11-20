#!/usr/bin/env python3
"""Held-environment verification of articulated video against matched capacity."""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from shuwei_c import q1v2_quality as q
from shuwei_c import q1v3_articulated_risk as learn
from shuwei_c.q1v3_shared_risk import features
from run_c_q1_v3_articulated_risk import select_visual
from run_c_q1_v3_quality_oof import inner_prior,save_predictions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--horizon',type=float,choices=[.1,.5],default=.1);parser.add_argument('--prior',choices=['transition','frequency'],default='transition');args=parser.parse_args()
    baseline_name=args.prior;transition=args.prior=='transition'
    tag='' if args.horizon==.1 and transition else f'_h{args.horizon}_{args.prior}'
    out=Path('artifacts/c_q1_v3/articulated_oof'+tag);out.mkdir(exist_ok=True);start=time.perf_counter()
    if (out/'manifest.json').exists():raise RuntimeError('Completed experiment exists; do not overwrite')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
    train=pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_train_{args.horizon}.parquet')
    val=pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_validation_{args.horizon}.parquet')
    raw={};numerical,flags,_=features(train,None,False)
    visual=np.load('artifacts/c_q1_v3/articulated_features/quality_train.npz')['features']
    def aligned(split,frame):
        source=pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_{split}_0.1.parquet')
        ix=pd.MultiIndex.from_frame(source[['episode','origin']]).get_indexer(pd.MultiIndex.from_frame(frame[['episode','origin']]))
        assert (ix>=0).all();np.testing.assert_array_equal(source.time_s.to_numpy()[ix],frame.time_s.to_numpy())
        return np.load(f'artifacts/c_q1_v3/articulated_features/quality_{split}.npz')['features'][ix]
    if args.horizon!=.1:visual=aligned('train',train)
    kinds=['numeric','scalar','shape','motion'];seeds=[20261120,20261121,20261122]
    for kind in kinds:raw[kind]=np.concatenate([numerical,flags[...,None],*([] if kind=='numeric' else [select_visual(visual,kind)])],axis=-1)
    oof={name:np.zeros((len(train),4)) for name in [baseline_name,*[f'{k}_r{r}' for k in kinds for r in [.001,.01]]]};folds=[]
    def prior_for(frame):
        prior=np.zeros((len(frame),4))
        for fi,hi in GroupKFold(3).split(frame,groups=frame.environment):prior[hi]=q.predict_baseline(q.fit_baseline(frame.iloc[fi],transition),frame.iloc[hi])
        return prior
    for fold,(fi,hi) in enumerate(GroupKFold(4).split(train,groups=train.environment)):
        fitting=train.iloc[fi];held=train.iloc[hi];fp=prior_for(fitting);hp=q.predict_baseline(q.fit_baseline(fitting,transition),held);oof[baseline_name][hi]=hp
        for kind in kinds:
            for reg in [.001,.01]:
                name=f'{kind}_r{reg}';values=[]
                for seed in seeds:
                    m=learn.fit(fitting,raw[kind][fi],fp,reg,seed);torch.save(m,out/f'fold{fold}_{name}_s{seed}.pt');values.append(learn.predict(raw[kind][hi],hp,m))
                oof[name][hi]=np.mean(values,axis=0)
        scores={name:next(r['brier'] for r in q.metrics(held,p[hi],name) if r['dataset']=='ALL' and r['target']=='any' and r['condition']=='all') for name,p in oof.items()}
        folds.append(dict(fold=fold,held_environments=sorted(held.environment.unique()),scores=scores));print(json.dumps(folds[-1]),flush=True)
        save_predictions(train,oof,out/'train_oof.partial.parquet')
    save_predictions(train,oof,out/'train_oof.parquet');vp={baseline_name:q.predict_baseline(q.fit_baseline(train,transition),val)}
    if not tag:
        cache=np.load('artifacts/c_q1_v3/articulated_risk/ensemble.npz');vp.update({name:cache[name+'_e20'] for name in oof if name!=baseline_name})
    else:
        vn,vf,_=features(val,None,False);vv=aligned('validation',val);fp=oof[baseline_name]
        for kind in kinds:
            vx=np.concatenate([vn,vf[...,None],*([] if kind=='numeric' else [select_visual(vv,kind)])],axis=-1)
            for reg in [.001,.01]:
                name=f'{kind}_r{reg}';values=[]
                for seed in seeds:
                    m=learn.fit(train,raw[kind],fp,reg,seed);torch.save(m,out/f'full_{name}_s{seed}.pt');values.append(learn.predict(vx,vp[baseline_name],m))
                vp[name]=np.mean(values,axis=0)
    rows=[]
    for split,f,probs in [('train_oof',train,oof),('validation',val,vp)]:
        for name,p in probs.items():
            values=q.metrics(f,p,name)
            for r in values:r['split']=split
            rows+=values
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);save_predictions(val,vp,out/'validation.parquet')
    (out/'folds.json').write_text(json.dumps(folds,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(dict(test_used=False,horizon=args.horizon,prior=baseline_name,elapsed_s=time.perf_counter()-start,scope='Outer held environments excluded from network, standardization and all inner priors; development-selected candidate family',source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_articulated_risk.py')]}),indent=2)+'\n')


if __name__=='__main__':main()
