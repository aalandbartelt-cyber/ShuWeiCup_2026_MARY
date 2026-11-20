"""Causal alignment of frozen visual features; PCA fitted on training videos only.

AI-assisted by Codex, 2026-11-20. Wrist crop slots are deliberately unordered.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA

VIEWS=['head_left','head_right','left_wrist','right_wrist']


def causal_indices(times,queries):
    times=np.asarray(times);queries=np.asarray(queries)
    index=np.searchsorted(times,queries,side='right')-1
    known=index>=0;safe=np.maximum(index,0)
    age=np.where(known,queries-times[safe],0.)
    assert np.all(age>=-1e-9)
    return safe,known,age


def fit_pca(root,episodes,components=32,seed=20261120):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());rng=np.random.default_rng(seed);parts=[]
    for item in manifest:
        if item['episode'] not in episodes:continue
        data=np.load(root/f"{item['episode']}__{item['view']}.npz")
        values=data['embeddings'][data['available']].astype(np.float32)
        if len(values)>400:values=values[rng.choice(len(values),400,replace=False)]
        parts.append(values)
    pca=PCA(n_components=components,svd_solver='randomized',random_state=seed)
    pca.fit(np.concatenate(parts));return pca


def align_visual(frame,root,pca):
    root=Path(root);parts={name:[] for name in ['diagnostics','appearance','motion']};names={name:[] for name in parts}
    # Preserve the row ordering used by the numerical expert pack.
    for view in VIEWS:
        dim=pca.n_components_;current=np.zeros((len(frame),2*dim),np.float32)
        history=np.zeros((len(frame),4*dim),np.float32);diag=np.zeros((len(frame),32*3),np.float32)
        for episode,ix in frame.groupby('episode',sort=False).groups.items():
            file=root/f'{episode}__{view}.npz'
            if not file.exists():continue
            data=np.load(file);times=data['time_s'];emb=data['embeddings'].astype(np.float32)
            projected=pca.transform(emb.reshape(-1,emb.shape[-1])).reshape(len(times),3,-1)
            projected*=data['available'][...,None]
            g=frame.loc[ix];side=g.hand.to_numpy().astype(int);origins=g.origin_s.to_numpy()
            snapshots=[];diagnostics=[]
            for lag in [0.,.2,.4]:
                j,known,age=causal_indices(times,origins-lag);d=data['diagnostics'][j];available=data['available'][j]
                if view.startswith('head'):
                    own=projected[j,side+1];own_stats=np.stack([d[k,(s+1)*12:(s+2)*12] for k,s in enumerate(side)])
                    assignment=np.stack([d[k,36+s*4:40+s*4] for k,s in enumerate(side)])
                else:
                    count=np.maximum(available[:,1:].sum(axis=1),1)
                    own=projected[j,1:].sum(axis=1)/count[:,None]
                    own_stats=d[:,12:36].reshape(-1,2,12).sum(axis=1)/count[:,None]
                    assignment=d[:,36:].reshape(-1,2,4).mean(axis=1)
                feature=np.concatenate([projected[j,0],own],axis=1)*known[:,None]
                diagnostic=np.c_[d[:,:12],own_stats,assignment,known,age,available[:,1:].sum(axis=1),np.full(len(j),1.)]
                diagnostic[~known]=0.
                snapshots.append(feature);diagnostics.append(diagnostic)
            current[ix]=snapshots[0]
            history[ix]=np.concatenate([snapshots[0]-snapshots[1],snapshots[0]-snapshots[2]],axis=1)
            diag[ix]=np.concatenate([diagnostics[0],diagnostics[1],diagnostics[2]],axis=1)
        parts['appearance'].append(current);parts['motion'].append(history);parts['diagnostics'].append(diag)
        for name,array in [('appearance',current),('motion',history),('diagnostics',diag)]:
            names[name]+=[f'{view}_{name}_{i}' for i in range(array.shape[1])]
    return {key:np.concatenate(values,axis=1) for key,values in parts.items()},names
