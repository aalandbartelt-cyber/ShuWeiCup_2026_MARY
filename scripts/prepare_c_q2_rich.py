#!/usr/bin/env python3
"""Extend existing Q2 cache using the same disclosed-history replay and keys."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset
from shuwei_c.q2_replay import Window,Periodic,replay
from shuwei_c.q2_predictors import Kinematic
from shuwei_c.q2_rich_features import RichExperts,EXPERT_NAMES

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','validation'],required=True)
    ap.add_argument('--out',type=Path,default=Path('artifacts/c_q2/rich_state/data'));a=ap.parse_args()
    out=a.out/a.split;out.mkdir(parents=True,exist_ok=True)
    old=Path('artifacts/c_q2/learning_data')/a.split;config=json.loads(Path('configs/c_q2.json').read_text())
    sources=[Path(__file__).resolve().relative_to(Path.cwd()),Path('src/shuwei_c/q2_rich_features.py'),Path('src/shuwei_c/q2_features.py'),Path('src/shuwei_c/q2_replay.py'),old/'manifest.json',Path('artifacts/c_q2/protocol/development_windows.csv')]
    identity=dict(split=a.split,sources={str(p):sha(p) for p in sources},test_used=False,experts=EXPERT_NAMES)
    if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==identity
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    for p in sources:
        target=out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    windows=pd.read_csv('artifacts/c_q2/protocol/development_windows.csv');windows=windows[windows.split.eq(a.split)]
    data=Dataset();start=time.perf_counter();completed=[]
    for episode,g in windows.groupby('episode',sort=False):
        marker=out/f'{episode}.json'
        if marker.exists():
            saved=json.loads(marker.read_text());assert sha(out/f'{episode}.npz')==saved['npz_sha256'];completed.append(saved);continue
        rec=data.record(episode);numerical=rec.hands();features=[];experts=[];anchors=[];keys=[];names=None
        for row in g.itertuples(index=False):
            w=Window(row.window_id,episode,rec.dataset,rec.environment,rec.split,int(row.origin_index),int(row.stop),row.origin_s,row.horizon_s)
            for cap in config['request_caps']:
                for delay in ([.1] if cap==0 else [.1,0.,.3]):
                    scenario=f'b{cap}_d{delay:g}';seed=int(hashlib.sha256((w.window_id+scenario).encode()).hexdigest()[:8],16)
                    phase=float(np.random.default_rng(seed).choice(config['periodic_phases'])) if a.split=='train' else .5
                    extractor=RichExperts(rec.dataset)
                    def observe(ctx,pred):
                        nonlocal names
                        pack=extractor.at(ctx.history,ctx.now);features.append(pack['features']);experts.append(pack['experts']);anchors.append(pack['anchor']);names=pack['feature_names']
                        keys.extend((w.window_id,cap,delay,phase,ctx.now,hand) for hand in range(2))
                    replay(rec,w,Kinematic(),Periodic(cap,phase),cap,delay,observer=observe,numerical=numerical)
        f=pd.read_parquet(old/f'{episode}.parquet');old_arrays=np.load(old/f'{episode}.npz')
        features=np.concatenate(features);experts=np.concatenate(experts);anchors=np.concatenate(anchors).astype(np.float32)
        generated=pd.DataFrame(keys,columns=['window_id','cap','delay_s','phase','time_s','hand'])
        pd.testing.assert_frame_equal(generated.reset_index(drop=True),f[generated.columns].reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=0.)
        assert np.array_equal(anchors,old_arrays['anchor'])
        assert np.array_equal(experts[:,:,:16],old_arrays['experts'])
        assert np.array_equal(features[:,:old_arrays['features'].shape[1]],old_arrays['features'])
        dest=out/f'{episode}.npz';np.savez_compressed(dest,features=features,experts=experts,target=old_arrays['target'],anchor=anchors)
        f.to_parquet(out/f'{episode}.parquet',index=False)
        info=dict(episode=episode,rows=len(f),dataset=rec.dataset,feature_names=names,expert_names=EXPERT_NAMES,npz_sha256=sha(dest),prefix_exactly_matches_original=True,elapsed_s=time.perf_counter()-start)
        marker.write_text(json.dumps(info,indent=2)+'\n');completed.append(info);print(json.dumps({k:info[k] for k in ['episode','rows','elapsed_s']}),flush=True)
    manifest=dict(**identity,records=len(completed),rows=sum(r['rows'] for r in completed),feature_names=completed[0]['feature_names'],expert_names=EXPERT_NAMES,elapsed_s=time.perf_counter()-start)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
