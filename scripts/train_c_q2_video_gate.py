#!/usr/bin/env python3
"""Matched numerical/image gates with an untouched outer environment fold."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.data import Dataset
from shuwei_c.motion import estimate_noise
from shuwei_c.q2_learning import load_pack, subset, score_predictions
from shuwei_c.q2_video_learning import augment
from shuwei_c.q2_video_provider import VerifiedVideoCache


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('artifacts/c_q2/video_point_probe'))
    parser.add_argument('--cache',type=Path,default=Path('artifacts/c_q2/video/image_motion'))
    parser.add_argument('--kind',choices=['image','tracker'],default='image')
    parser.add_argument('--time-mode',default='nominal')
    parser.add_argument('--views',nargs='+',choices=['head_left','head_right','left_wrist','right_wrist'],
                        default=['head_left','head_right','left_wrist','right_wrist'])
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--seeds',nargs='+',type=int,default=[20261120])
    parser.add_argument('--per-group',type=int,default=16)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
    script=Path('scripts/train_c_q2_static.py')
    spec=importlib.util.spec_from_file_location('static_trainer',script)
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    files=[Path(__file__),script,Path('src/shuwei_c/q2_video_learning.py'),
           Path('src/shuwei_c/q2_video_provider.py'),Path('src/shuwei_c/q2_learning.py'),
           Path('src/shuwei_c/q2_features.py'),Path('src/shuwei_c/q2_sequence.py'),
           args.cache/'manifest.json',Path('artifacts/c_q2/protocol/folds.json')]
    identity=dict(kind=args.kind,time_mode=args.time_mode,views=args.views,fold=args.fold,seeds=args.seeds,
                  per_group=args.per_group,batch=256,max_updates=1200,patience_checks=8,
                  input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  test_used=False,official_validation_used=False,
                  controls='shifted and permuted held features under aligned-image weights')
    ip=args.out/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:
        raise ValueError('Experiment changed; use new directory')
    ip.write_text(json.dumps(identity,indent=2)+'\n')
    for p in files:
        relative=p.relative_to(Path.cwd()) if p.is_absolute() else p
        dst=args.out/'source_snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    folds=json.loads(Path('artifacts/c_q2/protocol/folds.json').read_text());outer=folds[args.fold]
    inner_held=folds[(args.fold+1)%4]['held_environments']
    inner_fit=sorted(set(outer['training_environments'])-set(inner_held))
    data=Dataset();noise=estimate_noise([r for r in data.records('train') if r.environment in inner_fit])
    raw=load_pack('artifacts/c_q2/learning_data/train');cache=VerifiedVideoCache(args.cache,kind=args.kind)
    masks=dict(inner_fit=trainer.choose_rows(raw['meta'],inner_fit,args.per_group),
               inner_held=np.flatnonzero(raw['meta'].environment.isin(inner_held).to_numpy()&(raw['meta'].horizon_s<=1).to_numpy()),
               outer_fit=trainer.choose_rows(raw['meta'],outer['training_environments'],args.per_group),
               outer_held=np.flatnonzero(raw['meta'].environment.isin(outer['held_environments']).to_numpy()))
    np.savez_compressed(args.out/'row_indices.npz',**masks)
    summary=[];started=time.perf_counter()
    for branch in ('numeric','image'):
        folder=args.out/branch;folder.mkdir(exist_ok=True)
        packs={k:subset(raw,ix) for k,ix in masks.items()}
        if branch=='image':
            packs={k:augment(p,cache,time_mode=args.time_mode,views=args.views) for k,p in packs.items()}
        predictions=[]
        for seed in args.seeds:
            chosen=trainer.fit(packs['inner_fit'],noise,seed,256,1200,packs['inner_held'],25,8)
            trainer.save_model(folder/f'inner_seed{seed}.pt',chosen,dict(inner_fit=inner_fit,inner_held=inner_held))
            model=trainer.fit(packs['outer_fit'],outer['noise'],seed,256,chosen['selected_updates'])
            trainer.save_model(folder/f'seed{seed}.pt',model,dict(outer_held=outer['held_environments']))
            prediction=trainer.predict(packs['outer_held'],model);predictions.append(prediction)
            frame,table=score_predictions(packs['outer_held'],prediction,outer['noise'])
            table['branch']=branch;table['seed']=seed;table['fold']=args.fold;summary.append(table)
            frame.to_parquet(folder/f'seed{seed}_errors.parquet',index=False)
            if branch=='image':
                for tag,extra,control in [('delayed',.3,'aligned'),('permuted',0.,'permuted')]:
                    held=augment(subset(raw,masks['outer_held']),cache,time_mode=args.time_mode,
                                 extra_delay=extra,control=control,views=args.views)
                    pred=trainer.predict(held,model);f,t=score_predictions(held,pred,outer['noise'])
                    t['branch']=tag;t['seed']=seed;t['fold']=args.fold;summary.append(t)
                    f.to_parquet(folder/f'seed{seed}_{tag}_errors.parquet',index=False)
                    del held
            print(json.dumps(dict(branch=branch,seed=seed,selected_updates=chosen['selected_updates'],elapsed_s=time.perf_counter()-started)),flush=True)
        del packs
    held=subset(raw,masks['outer_held']);_,table=score_predictions(held,held['experts'][:,:,8],outer['noise'])
    table['branch']='fast_lag_damped';table['seed']=-1;table['fold']=args.fold;summary.append(table)
    pd.concat(summary,ignore_index=True).to_csv(args.out/'comparison.csv',index=False)
    (args.out/'manifest.json').write_text(json.dumps(dict(**identity,elapsed_s=time.perf_counter()-started,status='training-only visual increment probe'),indent=2)+'\n')


if __name__=='__main__':
    main()
