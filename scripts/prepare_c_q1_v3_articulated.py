#!/usr/bin/env python3
"""Align articulated video and stereo anchors to existing numerical/quality rows."""
from pathlib import Path
import json,sys,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q1v3_landmarks import align_landmarks,align_stereo,NAMES
from shuwei_c.data import Dataset


def main():
    root=Path('artifacts/c_q1_v3/articulated_video');assert (root/'manifest.json').exists();out=Path('artifacts/c_q1_v3/articulated_features');out.mkdir(exist_ok=True);data=Dataset();start=time.perf_counter()
    for split in ['train','validation']:
        frame=pd.read_parquet(f'artifacts/c_q1_v3/state_screen/{split}_0.1_meta.parquet');x,names=align_landmarks(frame,root);p,quality=align_stereo(frame,root);correction=np.zeros_like(p)
        for episode,ix in frame.groupby('episode',sort=False).groups.items():
            t,points,_=data.record(episode).hands();center=points.mean(axis=2);g=frame.loc[ix];i=g.origin.to_numpy().astype(int);side=g.hand.to_numpy().astype(int)
            velocity=(center[i,side]-center[i-1,side])/np.maximum(t[i]-t[i-1],1e-9)[:,None]
            correction[ix]=(p[ix]+quality[ix,1,None]*velocity-center[i,side])/g.scale.to_numpy()[:,None]
        correction[quality[:,0]==0]=0.;correction=np.clip(correction,-10,10)
        np.savez_compressed(out/f'state_{split}.npz',features=x,anchor_correction=correction,anchor_quality=quality)
        q=pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_{split}_0.1.parquet');hands=[]
        for hand in [0,1]:
            meta=q[['episode','time_s']].rename(columns={'time_s':'origin_s'});meta['hand']=hand;z,_=align_landmarks(meta,root);hands.append(z)
        np.savez_compressed(out/f'quality_{split}.npz',features=np.stack(hands,axis=1))
        print(json.dumps(dict(split=split,state=x.shape,quality=(len(q),2,x.shape[1]),seconds=time.perf_counter()-start)),flush=True)
    (out/'feature_names.json').write_text(json.dumps(names,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,elapsed_s=time.perf_counter()-start,descriptor_names=NAMES,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_landmarks.py')]}),indent=2)+'\n')


if __name__=='__main__':main()
