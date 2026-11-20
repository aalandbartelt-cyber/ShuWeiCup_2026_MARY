#!/usr/bin/env python3
"""Train-selected camera-convention diagnostics against automatic 2D detections.

AI-assisted by Codex, 2026-11-20. Automatic detector agreement is not independent
3D truth. Unknown/missing calibration is retained explicitly.
"""
from pathlib import Path
import itertools,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from shuwei_c.data import Dataset
from shuwei_c.q1v2_state import nearest_indices

OUT=Path('artifacts/c_q1_v2');VIDEO=Path('artifacts/c_video')


def intrinsics(rec,view,times):
    if rec.dataset=='EgoDemo':
        meta=pd.read_parquet(rec.folder/'meta/episodes/chunk-000/file-000.parquet')
        a=np.array(meta['camera_intrinsics/'+view].iloc[0],float)
        return np.tile(a[:4],(len(times),1)),np.zeros(len(times)),a[4:]
    f=pd.read_csv(rec.folder/'tables'/f'sensor_camera_{view}_intrinsic.csv')
    ix=np.searchsorted(f._offset_from_first_log_s.to_numpy(),times,side='right')-1
    if (ix<0).any():raise ValueError('Intrinsics not available at first frame')
    k=f.iloc[ix]
    return k[['K[0]','K[4]','K[2]','K[5]']].to_numpy(),times-k._offset_from_first_log_s.to_numpy(),np.array([float(f['D.length'].max())])


def project(points,position,quaternion,intrinsic,convention):
    q=quaternion if convention['quaternion']=='xyzw' else quaternion[..., [1,2,3,0]]
    r=Rotation.from_quat(q).as_matrix()
    if convention['transpose']:r=r.transpose(0,2,1)
    relative=points-position[:,None,:]
    local=np.einsum('nhi,nij->nhj',relative,r)
    z=local[...,2]*convention['forward_sign']
    with np.errstate(divide='ignore',invalid='ignore'):
        x=convention['x_sign']*local[...,0]/z*intrinsic[:,0,None]+intrinsic[:,2,None]
        y=convention['y_sign']*local[...,1]/z*intrinsic[:,1,None]+intrinsic[:,3,None]
    return np.stack([x,y],axis=-1),z>1e-6


def main():
    data=Dataset();allrecords=data.records('train')+data.records('validation')
    candidates=[dict(quaternion=q,transpose=tr,forward_sign=z,x_sign=x,y_sign=y)
        for q,tr,z,x,y in itertools.product(['wxyz','xyzw'],[False,True],[1,-1],[1,-1],[1,-1])]
    prepared=[];alignment=[]
    for rec in allrecords:
        t,p,bad=rec.hands();centres=p.mean(axis=2)
        for view in ['head_left','head_right']:
            path=VIDEO/f'{rec.episode}__{view}.jsonl'
            raw=[json.loads(line) for line in path.open()];vt=np.array([r['time_s'] for r in raw])
            ix=nearest_indices(t,vt);camera=rec.streams[view+'_camera'];ci=nearest_indices(camera['time'],vt)
            position=camera['position'][ci].reshape(-1,3);q=camera['quaternion'][ci].reshape(-1,4)
            k,age,distortion=intrinsics(rec,view,vt)
            close=abs(t[ix]-vt)<=.55*np.median(np.diff(t))+1e-9
            alignment.append(dict(dataset=rec.dataset,split=rec.split,episode=rec.episode,view=view,
                images=len(vt),matched=int(close.sum()),median_abs_time_error=float(np.median(abs(t[ix]-vt))),
                maximum_abs_time_error=float(np.max(abs(t[ix]-vt))),distortion_nonzero=bool(np.any(distortion!=0))))
            prepared.append((rec,view,raw,centres[ix],bad[ix],position,q,k,close))
    pd.DataFrame(alignment).to_csv(OUT/'video_numerical_alignment.csv',index=False)
    selection=[];selected={}
    for dataset in ['EgoDemo','EgoPro','EgoStandard']:
        for number,convention in enumerate(candidates):
            per_record=[];matched=0;eligible=0
            for rec,view,raw,c,bad,position,q,k,close in prepared:
                if rec.split!='train' or rec.dataset!=dataset:continue
                uv,front=project(c,position,q,k,convention);errors=[]
                for i,row in enumerate(raw):
                    # Both detected hands and both qualified references avoid a
                    # convention winning simply by projecting every hand behind camera.
                    if not close[i] or bad[i].any() or len(row['detections'])!=2:continue
                    eligible+=1
                    detection=np.array([np.array(d['landmarks'])[:,:2].mean(axis=0) for d in row['detections']])
                    normalized=uv[i]/np.array([row['width'],row['height']])
                    if not front[i].all() or not np.isfinite(normalized).all():errors.append(2.);continue
                    cost=np.linalg.norm(normalized[:,None,:]-detection[None,:,:],axis=-1)
                    a,b=linear_sum_assignment(cost);errors.append(float(np.minimum(cost[a,b],2).mean()));matched+=1
                if errors:per_record.append(dict(environment=rec.environment,episode=rec.episode,error=float(np.median(errors))))
            f=pd.DataFrame(per_record)
            score=float(f.groupby(['environment','episode']).error.mean().groupby('environment').mean().mean())
            selection.append(dict(dataset=dataset,candidate=number,score=score,eligible=eligible,front=matched,**convention))
        best=min([r for r in selection if r['dataset']==dataset],key=lambda r:r['score'])
        selected[dataset]={k:best[k] for k in candidates[0]}
    pd.DataFrame(selection).to_csv(OUT/'camera_convention_candidates.csv',index=False)
    (OUT/'camera_conventions.json').write_text(json.dumps(dict(selected=selected,
        selection='training detections only; conditional two-hand detector agreement; not physical truth'),indent=2)+'\n')
    validation=[]
    for rec,view,raw,c,bad,position,q,k,close in prepared:
        uv,front=project(c,position,q,k,selected[rec.dataset])
        for i,row in enumerate(raw):
            if not close[i] or bad[i].any() or len(row['detections'])!=2:continue
            detected=np.array([np.array(d['landmarks'])[:,:2].mean(axis=0) for d in row['detections']])
            normalized=uv[i]/np.array([row['width'],row['height']]);cost=np.linalg.norm(normalized[:,None,:]-detected[None,:,:],axis=-1)
            if not np.isfinite(cost).all():continue
            a,b=linear_sum_assignment(cost)
            validation.append(dict(dataset=rec.dataset,split=rec.split,environment=rec.environment,episode=rec.episode,
                view=view,time_s=row['time_s'],image_width=row['width'],image_height=row['height'],
                projected_front=bool(front[i].all()),normalized_error=float(cost[a,b].mean()),
                source_frame=row['frame_index'],hand0_x=float(uv[i,0,0]),hand0_y=float(uv[i,0,1]),
                hand1_x=float(uv[i,1,0]),hand1_y=float(uv[i,1,1])))
    f=pd.DataFrame(validation);f.to_csv(OUT/'camera_projection_diagnostics.csv',index=False)
    print(json.dumps(selected),flush=True)
    print(f.groupby(['dataset','split']).normalized_error.agg(['count','median','mean']).to_string(),flush=True)


if __name__=='__main__':main()
