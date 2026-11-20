"""Articulated image geometry, causal histories and stereo ray intersection.

AI-assisted by Codex, 2026-11-20. Image landmark confidence is not accuracy.
"""
from pathlib import Path
import numpy as np
from .q1v3_visual import causal_indices

SCALARS=['detected','association_close','handedness_score','center_x','center_y','projection_dx','projection_dy','palm_span','axis_x','axis_y','image_edge','log_area']
NAMES=SCALARS+[f'canonical_joint_{j}_{a}' for j in range(21) for a in ['along','across']]+[f'bend_{f}_{k}' for f in range(5) for k in range(3)]+[f'tip_distance_{a}_{b}' for a in range(5) for b in range(a+1,5)]


def descriptor(points,expected,width,height,confidence):
    p=np.asarray(points)[:,:2]*[width,height];center=p.mean(axis=0);axis=p[9]-p[0];length=max(np.linalg.norm(axis),1.)
    u=axis/length;v=np.array([-u[1],u[0]]);canonical=np.c_[(p-p[0])@u,(p-p[0])@v]/length
    offset=(center-expected)/[width,height];distance=np.linalg.norm(offset)
    base=[1.,float(distance<.2),confidence,*(center/[width,height]),*offset,length/np.sqrt(width*height),*u,
        min((p/[width,height]).min(),(1-p/[width,height]).min()),np.log(max(np.prod(np.ptp(p,axis=0))/(width*height),1e-8))]
    bends=[]
    for finger in range(5):
        chain=[0,*range(1+4*finger,5+4*finger)];bone=np.diff(p[chain],axis=0)
        for k in range(3):bends.append(np.dot(bone[k],bone[k+1])/(np.linalg.norm(bone[k])*np.linalg.norm(bone[k+1])+1e-8))
    tips=p[[4,8,12,16,20]];distances=[np.linalg.norm(tips[a]-tips[b])/length for a in range(5) for b in range(a+1,5)]
    value=np.r_[base,np.clip(canonical.ravel(),-10,10),bends,np.clip(distances,0,10)]
    assert len(value)==len(NAMES) and np.isfinite(value).all()
    return value,center


def ray_midpoint(c1,d1,c2,d2):
    d1=d1/np.linalg.norm(d1,axis=-1,keepdims=True);d2=d2/np.linalg.norm(d2,axis=-1,keepdims=True)
    dot=np.sum(d1*d2,axis=-1);den=1-dot**2;offset=c2-c1
    a=np.sum(d1*offset,axis=-1);b=-np.sum(d2*offset,axis=-1)
    safe=np.maximum(den,1e-12);s=(a+dot*b)/safe;t=(dot*a+b)/safe
    p1=c1+s[...,None]*d1;p2=c2+t[...,None]*d2
    gap=np.linalg.norm(p1-p2,axis=-1);valid=(den>1e-4)&(s>0)&(t>0)
    return (p1+p2)/2,valid,np.sqrt(np.maximum(den,0)),gap


def align_landmarks(frame,root):
    root=Path(root);parts=[];names=[]
    for view in ['head_left','head_right']:
        out=np.zeros((len(frame),len(NAMES)*3+6),np.float32)
        for episode,ix in frame.groupby('episode',sort=False).groups.items():
            a=np.load(root/f'{episode}__{view}.npz');g=frame.loc[ix];side=g.hand.to_numpy().astype(int);now=g.origin_s.to_numpy();times=a['time_s']
            j,known,age=causal_indices(times,now);known &=age<=.25
            current=a['features'][j,side].copy();current[~known]=0;blocks=[current];flags=[known.astype(float),age]
            for lag in [2/15,4/15]:
                old,k,pastage=causal_indices(times,now-lag);k &=pastage<=.25
                previous=a['features'][old,side];both=known&k&(current[:,0]>0)&(previous[:,0]>0)
                dt=times[j]-times[old];delta=np.zeros_like(current);delta[both]=(current[both]-previous[both])/np.maximum(dt[both,None],1e-6)
                blocks.append(np.clip(delta,-100,100));flags.extend([both.astype(float),np.where(k,pastage,0.)])
            out[ix]=np.column_stack([*blocks,*flags])
        parts.append(out);names.extend([f'{view}_{prefix}_{n}' for prefix in ['now','change_.133','change_.267'] for n in NAMES]+[f'{view}_{n}' for n in ['available','age','history1','age1','history2','age2']])
    return np.concatenate(parts,axis=1),names


def align_stereo(frame,root):
    root=Path(root);position=np.zeros((len(frame),3));quality=np.zeros((len(frame),5))
    for episode,ix in frame.groupby('episode',sort=False).groups.items():
        a=np.load(root/f'{episode}__stereo.npz');g=frame.loc[ix];side=g.hand.to_numpy().astype(int);j,known,age=causal_indices(a['time_s'],g.origin_s.to_numpy())
        known &=age<=.15;valid=known&a['valid'][j,side];position[ix]=np.where(valid[:,None],a['position'][j,side],0.)
        quality[ix]=np.c_[valid,age,a['parallax'][j,side],a['ray_gap'][j,side],a['time_gap'][j]]
        quality[ix[~known]]=0.
    return position,quality
