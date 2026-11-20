#!/usr/bin/env python3
"""Grouped causal-history GRU ablations, OOF selection followed by validation."""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from shuwei_c import q1v2_quality as q
from shuwei_c import q1v3_sequence as learn
from shuwei_c.q1v3_shared_risk import features
from run_c_q1_v3_articulated_risk import select_visual
from run_c_q1_v3_quality_oof import save_predictions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--horizon',type=float,choices=[.1,.5],default=.1);args=parser.parse_args()
    out=Path(f'artifacts/c_q1_v3/sequence_h{args.horizon}');out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Completed experiment exists')
    start=time.perf_counter();source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_sequence.py')]}
    for p in source_hashes:(out/Path(p).name).write_bytes(Path(p).read_bytes())
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.48);frames={};sources={};raw={}
    for split in ['train','validation']:
        frames[split]=pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_{split}_{args.horizon}.parquet')
        sources[split]=pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_{split}_0.1.parquet')
        diag=np.load(f'artifacts/c_q1_v3/video_quality/visual_{split}.npz')['diagnostics']
        n,flag,_=features(sources[split],None,False);d,_,_=features(sources[split],diag,True)
        shape=np.load(f'artifacts/c_q1_v3/articulated_features/quality_{split}.npz')['features']
        raw[split]={'numeric':np.concatenate([n,flag[...,None]],axis=-1),'diagnostics':np.concatenate([d,flag[...,None]],axis=-1),
                    'shape':np.concatenate([n,flag[...,None],select_visual(shape,'shape')],axis=-1)}
    train=frames['train'];val=frames['validation'];transition=args.horizon==.1;baseline='transition' if transition else 'frequency'
    folds=list(GroupKFold(4).split(train,groups=train.environment));priors=[];full=np.zeros((len(train),4))
    for fi,hi in folds:
        f=train.iloc[fi];p=np.zeros((len(f),4))
        for a,b in GroupKFold(3).split(f,groups=f.environment):p[b]=q.predict_baseline(q.fit_baseline(f.iloc[a],transition),f.iloc[b])
        hp=q.predict_baseline(q.fit_baseline(f,transition),train.iloc[hi]);full[hi]=hp;priors.append((p,hp))
    vp=q.predict_baseline(q.fit_baseline(train,transition),val);oof={baseline:full};validation={baseline:vp};rows=[];seeds=[20261120,20261121,20261122]
    for kind in ['numeric','diagnostics','shape']:
        x,known=learn.history(sources['train'],train,raw['train'][kind]);vx,vknown=learn.history(sources['validation'],val,raw['validation'][kind])
        for reg in [.001,.01]:
            name=f'{kind}_r{reg}';p=np.zeros((len(train),4))
            for fold,((fi,hi),(fp,hp)) in enumerate(zip(folds,priors)):
                values=[]
                for seed in seeds:
                    m=learn.fit(train.iloc[fi],x[fi],known[fi],fp,reg,seed);torch.save(m,out/f'fold{fold}_{name}_s{seed}.pt');values.append(learn.predict(x[hi],known[hi],hp,m))
                p[hi]=np.mean(values,axis=0);print(json.dumps(dict(method=name,fold=fold,elapsed_s=time.perf_counter()-start)),flush=True)
            oof[name]=p;values=[]
            for seed in seeds:
                m=learn.fit(train,x,known,full,reg,seed);torch.save(m,out/f'full_{name}_s{seed}.pt');values.append(learn.predict(vx,vknown,vp,m))
            validation[name]=np.mean(values,axis=0)
            for split,f,prob in [('train_oof',train,p),('validation',val,validation[name])]:
                values=q.metrics(f,prob,name)
                for row in values:row['split']=split
                rows+=values
            pd.DataFrame(rows).to_csv(out/'metrics.partial.csv',index=False);save_predictions(train,oof,out/'train_oof.partial.parquet');save_predictions(val,validation,out/'validation.partial.parquet')
    for split,f,p in [('train_oof',train,full),('validation',val,vp)]:
        values=q.metrics(f,p,baseline)
        for row in values:row['split']=split
        rows+=values
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);save_predictions(train,oof,out/'train_oof.parquet');save_predictions(val,validation,out/'validation.parquet')
    (out/'folds.json').write_text(json.dumps([dict(fold=i,held_environments=sorted(train.iloc[hi].environment.unique())) for i,(_,hi) in enumerate(folds)],indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,horizon=args.horizon,elapsed_s=time.perf_counter()-start,source_hashes=source_hashes,history='8 snapshots at 0.1-second spacing, oldest first, no lookahead; shared-hand GRU16; whole-environment outer folds with inner prior fitting',seeds=seeds,regularization=[.001,.01],epochs=20),indent=2)+'\n')


if __name__=='__main__':main()
