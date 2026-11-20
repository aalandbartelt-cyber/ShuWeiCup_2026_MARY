"""Head-view diagnostic alignment identical to development, without DINO/PCA."""
from pathlib import Path
import numpy as np
from .q1v3_visual import causal_indices


def align(frame,root):
    out=np.zeros((len(frame),4*3*32),np.float32);root=Path(root)
    for view_index,view in enumerate(['head_left','head_right']):
        for episode,ix in frame.groupby('episode',sort=False).groups.items():
            path=root/f'{episode}__{view}.npz'
            if not path.exists():raise FileNotFoundError(path)
            data=np.load(path);times=data['time_s'];g=frame.loc[ix];side=g.hand.to_numpy().astype(int);origin=g.origin_s.to_numpy();values=[]
            for lag in [0.,.2,.4]:
                j,known,age=causal_indices(times,origin-lag);d=data['diagnostics'][j];available=data['available'][j]
                own=np.stack([d[k,(s+1)*12:(s+2)*12] for k,s in enumerate(side)])
                assignment=np.stack([d[k,36+s*4:40+s*4] for k,s in enumerate(side)])
                diagnostic=np.c_[d[:,:12],own,assignment,known,age,available[:,1:].sum(axis=1),np.ones(len(j))];diagnostic[~known]=0.;values.append(diagnostic)
            out[ix,view_index*96:(view_index+1)*96]=np.concatenate(values,axis=1)
    return out


def joint(frame,root):
    hands=[]
    for hand in [0,1]:
        meta=frame[['episode','time_s']].rename(columns={'time_s':'origin_s'}).copy();meta['hand']=hand;hands.append(align(meta,root))
    return np.concatenate(hands,axis=1)
