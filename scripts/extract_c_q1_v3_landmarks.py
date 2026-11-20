#!/usr/bin/env python3
"""Use existing 15 Hz 2D detections to expose articulation and stereo anchors.

AI-assisted by Codex, 2026-11-20. No new detector fitting and no test data.
"""
from pathlib import Path
import json,sys,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from shuwei_c.data import Dataset
from shuwei_c.q1v2_state import nearest_indices
from shuwei_c.q1v3_landmarks import descriptor,NAMES,ray_midpoint
from audit_c_q1_v2_geometry import intrinsics


def main():
    out=Path('artifacts/c_q1_v3/articulated_video');out.mkdir(exist_ok=True);source=Path('artifacts/c_video_v2_15hz');data=Dataset();start=time.perf_counter();manifest=[];summary=[]
    for rec in data.records('train')+data.records('validation'):
        t,p,bad=rec.hands();centers=p.mean(axis=2);views={}
        for view in ['head_left','head_right']:
            rows=[json.loads(line) for line in (source/f'{rec.episode}__{view}.jsonl').open()];vt=np.array([r['time_s'] for r in rows]);ri=np.searchsorted(t,vt,side='right')-1
            camera=rec.streams[view+'_camera'];ci=np.searchsorted(camera['time'],vt,side='right')-1;assert (ci>=0).all()
            c=camera['position'][ci].reshape(-1,3);q=camera['quaternion'][ci].reshape(-1,4);rot=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix();k,_,distortion=intrinsics(rec,view,vt)
            local=np.einsum('nhi,nij->nhj',centers[np.maximum(ri,0)]-c[:,None,:],rot);safe=np.where(abs(local[...,2])>1e-8,local[...,2],1e-8)
            projected=np.stack([local[...,0]/safe*k[:,0,None]+k[:,2,None],local[...,1]/safe*k[:,1,None]+k[:,3,None]],axis=-1)
            features=np.zeros((len(rows),2,len(NAMES)),np.float32);rays=np.zeros((len(rows),2,3));rays[...,2]=1.;accepted=np.zeros((len(rows),2),bool)
            for i,row in enumerate(rows):
                detection=row['detections'];width,height=row['width'],row['height']
                if not detection or ri[i]<0:continue
                points=np.array([np.array(d['landmarks'])[:,:2].mean(axis=0)*[width,height] for d in detection])
                distance=np.linalg.norm((projected[i,:,None]-points[None,:,:])/[width,height],axis=-1);hands,det=linear_sum_assignment(distance)
                for hand,d in zip(hands,det):
                    features[i,hand],center=descriptor(detection[d]['landmarks'],projected[i,hand],width,height,detection[d]['handedness_score'])
                    direction=np.array([(center[0]-k[i,2])/k[i,0],(center[1]-k[i,3])/k[i,1],1.]);rays[i,hand]=rot[i]@direction
                    accepted[i,hand]=(distance[hand,d]<.2)&(local[i,hand,2]>0)
            views[view]=dict(time_s=vt,features=features,camera=c,rays=rays,accepted=accepted)
            np.savez_compressed(out/f'{rec.episode}__{view}.npz',**views[view]);manifest.append(dict(episode=rec.episode,split=rec.split,dataset=rec.dataset,view=view,frames=len(vt),distortion_nonzero=bool(np.any(distortion!=0))))
        left,right=views['head_left'],views['head_right'];j=nearest_indices(right['time_s'],left['time_s']);available=np.maximum(left['time_s'],right['time_s'][j]);timegap=abs(left['time_s']-right['time_s'][j])
        position,valid,parallax,gap=ray_midpoint(left['camera'][:,None,:],left['rays'],right['camera'][j,None,:],right['rays'][j])
        valid &=left['accepted']&right['accepted'][j]&(timegap[:,None]<.04)
        # Output timestamp is when both paired images have arrived.
        assert np.all(np.diff(available)>=0)
        np.savez_compressed(out/f'{rec.episode}__stereo.npz',time_s=available,position=position,valid=valid,parallax=parallax,ray_gap=gap,time_gap=timegap)
        ri=np.searchsorted(t,available,side='right')-1;qualified=valid&~bad[np.maximum(ri,0)]&(ri[:,None]>=0)
        error=np.linalg.norm(position-centers[np.maximum(ri,0)],axis=-1)
        summary.append(dict(episode=rec.episode,dataset=rec.dataset,environment=rec.environment,split=rec.split,frames=len(available),stereo_valid_fraction=float(valid.mean()),qualified_comparisons=int(qualified.sum()),median_reference_disagreement=float(np.median(error[qualified])) if qualified.any() else None,median_baseline=float(np.median(np.linalg.norm(left['camera']-right['camera'][j],axis=1)))))
        print(json.dumps(summary[-1]),flush=True)
    pd.DataFrame(summary).to_csv(out/'stereo_diagnostics.csv',index=False);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps(dict(feature_names=NAMES,test_used=False,elapsed_s=time.perf_counter()-start,limits='No distortion correction; 2D mean ray intersection is a noisy visual anchor, not physical ground truth. Projection association uses only available numerical history. Stereo availability is max of the two image times.',source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_landmarks.py')]}),indent=2)+'\n')


if __name__=='__main__':main()
