#!/usr/bin/env python3
"""Causal DINOv2 global/hand crops and image-motion diagnostics on the local GPU.

AI-assisted by Codex, 2026-11-20. Pretrained weights frozen; no labels used to
extract features. Uses previously audited 15 Hz frame indices, train/val only.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from shuwei_c.data import Dataset
from audit_c_q1_v2_geometry import intrinsics


def letterbox(image,size=224):
    h,w=image.shape[:2];scale=size/max(h,w);nh,nw=max(1,round(h*scale)),max(1,round(w*scale))
    canvas=np.full((size,size,3),127,np.uint8);resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
    y=(size-nh)//2;x=(size-nw)//2;canvas[y:y+nh,x:x+nw]=resized
    return cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)


def crop_box(points,width,height):
    points=np.asarray(points)
    if not len(points) or not np.isfinite(points).all():return None
    lo=points.min(axis=0);hi=points.max(axis=0);center=(lo+hi)/2
    if not (0<=center[0]<width and 0<=center[1]<height):return None
    side=float(np.clip(max(hi-lo)*1.6,64,max(width,height)*.6));x,y=center
    return np.array([max(0,int(x-side/2)),max(0,int(y-side/2)),min(width,int(x+side/2)),min(height,int(y+side/2))],int)


def projected_hands(rec,view,vt):
    t,p,_=rec.hands();i=np.searchsorted(t,vt,side='right')-1;known=i>=0;i=np.maximum(i,0)
    cam=rec.streams[view+'_camera'];ci=np.searchsorted(cam['time'],vt,side='right')-1
    if (ci<0).any():raise ValueError('Camera is not yet available')
    pos=cam['position'][ci].reshape(-1,3);q=cam['quaternion'][ci].reshape(-1,4);rot=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    xyz=np.einsum('nhki,nij->nhkj',p[i]-pos[:,None,None,:],rot);k,_,_=intrinsics(rec,view,vt)
    z=xyz[...,2];safe=np.where(abs(z)>1e-7,z,1e-7)
    x=xyz[...,0]/safe*k[:,0,None,None]+k[:,2,None,None];y=xyz[...,1]/safe*k[:,1,None,None]+k[:,3,None,None]
    points=np.stack([x,y],axis=-1);valid=(z>0).all(axis=-1)&known[:,None]
    return points,valid


def region_stats(gray,flow,box,background,dt):
    height,width=gray.shape
    if box is None:return np.zeros(12,np.float32)
    x0,y0,x1,y1=box;x0,x1=np.clip([x0,x1],0,width);y0,y1=np.clip([y0,y1],0,height)
    patch=gray[y0:y1,x0:x1]
    if patch.size==0:return np.zeros(12,np.float32)
    f=flow[y0:y1,x0:x1];motion=np.median(f.reshape(-1,2),axis=0);residual=f-background
    magnitude=np.linalg.norm(residual,axis=-1);edge=min(x0/width,y0/height,1-x1/width,1-y1/height)
    return np.array([1.,patch.mean()/255,patch.std()/255,np.log1p(cv2.Laplacian(patch,cv2.CV_32F).var()),
        np.mean((patch<8)|(patch>247)),patch.size/(width*height),edge,
        motion[0]/max(dt,1e-6)/width,motion[1]/max(dt,1e-6)/height,
        np.median(magnitude)/max(dt,1e-6)/width,np.quantile(magnitude,.9)/max(dt,1e-6)/width,float(dt>0)],np.float32)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit-videos',type=int);parser.add_argument('--batch',type=int,default=24)
    parser.add_argument('--worker-index',type=int,default=0);parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
    if not 0<=args.worker_index<args.workers:raise ValueError('Invalid worker partition')
    out=Path('artifacts/c_q1_v3/visual_dino15');out.mkdir(exist_ok=True);source=Path('artifacts/c_video_v2_15hz');manifest=json.loads((source/'video_manifest.json').read_text())
    manifest=sorted(manifest,key=lambda r:(r['split']!='train',r['dataset'],r['episode'],r['view']))
    if args.limit_videos:manifest=manifest[:args.limit_videos]
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.cuda.set_per_process_memory_fraction(.15)
    model=torch.hub.load('artifacts/external/dinov2','dinov2_vits14_reg',source='local',pretrained=True).eval().cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    checkpoint=Path.home()/'.cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth'
    identity=dict(backbone='dinov2_vits14_reg',repo_sha=subprocess.check_output(['git','-C','artifacts/external/dinov2','rev-parse','HEAD'],text=True).strip(),
        weight_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        temporal_basis='15 Hz audited timestamps; frames at or before query only; flow previous->current',
        views='head slots use current available 3D projection association; wrist slots ordered by image x, not anatomical labels',
        input_size=224,torch=torch.__version__,test_used=False)
    if args.worker_index==0:(out/'protocol.json').write_text(json.dumps(identity,indent=2)+'\n')
    data=Dataset();completed=[];started=time.perf_counter()
    for number,item in enumerate(manifest):
        if number%args.workers!=args.worker_index:continue
        uid=item['episode'];view=item['view'];target=out/f'{uid}__{view}.npz';meta_path=target.with_suffix('.json')
        if target.exists() and meta_path.exists():
            old=json.loads(meta_path.read_text())
            if old['identity']!=identity:raise ValueError('Cached visual identity mismatch')
            completed.append(old);continue
        rec=data.record(uid);raw=[json.loads(line) for line in (source/item['features_file']).open()];vt=np.array([r['time_s'] for r in raw])
        if view.startswith('head_'):projections,visible=projected_hands(rec,view,vt)
        cap=cv2.VideoCapture(str(data.root/item['source']));wanted={r['frame_index']:k for k,r in enumerate(raw)};frame_index=0;previous=None;previous_t=None
        embeddings=[];diagnostics=[];boxes_record=[];queue=[];slot_counts=[];batches=[]
        def flush():
            if not queue:return
            batch=torch.as_tensor(np.stack(queue),device='cuda').permute(0,3,1,2).float()/255
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):features=model((batch-mean)/std)
            embeddings.extend(features.cpu().numpy().astype(np.float16));queue.clear()
        while True:
            ok,image=cap.read()
            if not ok:break
            if frame_index not in wanted:frame_index+=1;continue
            index=wanted[frame_index];row=raw[index];height,width=image.shape[:2]
            detected=[np.array(d['landmarks'])[:,:2]*[width,height] for d in row['detections']];boxes=[None,None];slots=np.zeros((2,4),np.float32)
            if view.startswith('head_'):
                expected=projections[index].mean(axis=1)
                for hand in range(2):
                    if visible[index,hand]:boxes[hand]=crop_box(projections[index,hand],width,height)
                if detected:
                    centers=np.array([d.mean(axis=0) for d in detected]);distance=np.linalg.norm((expected[:,None]-centers[None,:,:])/[width,height],axis=-1)
                    ah,ad=linear_sum_assignment(distance)
                    for hand,detection in zip(ah,ad):
                        slots[hand]=[1,float(row['detections'][detection]['handedness_score']),min(distance[hand,detection],2.),len(detected)]
                        if distance[hand,detection]<.2:boxes[hand]=crop_box(detected[detection],width,height)
            else:
                for hand,detection in enumerate(sorted(range(len(detected)),key=lambda d:detected[d][:,0].mean())[:2]):
                    boxes[hand]=crop_box(detected[detection],width,height);slots[hand]=[1,float(row['detections'][detection]['handedness_score']),0.,len(detected)]
            all_boxes=[np.array([0,0,width,height]),*boxes]
            small=cv2.resize(image,(320,round(height*320/width)),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
            dt=0. if previous is None else float(vt[index]-previous_t)
            flow=np.zeros((*gray.shape,2),np.float32) if previous is None else cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,15,3,5,1.2,0)
            background=np.median(flow.reshape(-1,2),axis=0);ratio=np.array([gray.shape[1]/width,gray.shape[0]/height]*2)
            stats=[]
            for box in all_boxes:
                if box is None:queue.append(np.full((224,224,3),127,np.uint8));stats.extend(region_stats(gray,flow,None,background,dt))
                else:
                    x0,y0,x1,y1=box;queue.append(letterbox(image[y0:y1,x0:x1]));stats.extend(region_stats(gray,flow,np.rint(box*ratio).astype(int),background,dt))
                if len(queue)>=args.batch:flush()
            diagnostics.append(np.r_[stats,slots.ravel()]);boxes_record.append(np.array([np.full(4,-1) if b is None else b for b in all_boxes]));slot_counts.append([b is not None for b in all_boxes])
            previous=gray;previous_t=vt[index];frame_index+=1
        cap.release();flush();assert len(diagnostics)==len(raw) and len(embeddings)==3*len(raw)
        temporary=target.with_suffix('.tmp.npz');np.savez_compressed(temporary,time_s=vt,frame_index=np.array([r['frame_index'] for r in raw]),
            embeddings=np.array(embeddings).reshape(len(raw),3,-1),diagnostics=np.array(diagnostics),boxes=np.array(boxes_record),available=np.array(slot_counts));temporary.replace(target)
        metadata=dict(identity=identity,episode=uid,dataset=rec.dataset,split=rec.split,environment=rec.environment,view=view,frames=len(raw),
            diagnostics_region_fields=['available','brightness','contrast','log_laplacian_variance','clipped_intensity_fraction','area_fraction','boundary_distance','flow_x_per_s','flow_y_per_s','compensated_flow_median','compensated_flow_p90','flow_history_available'],
            diagnostics_assignment_fields=['detected','handedness_score_not_accuracy','projection_distance','detection_count'])
        meta_path.write_text(json.dumps(metadata,indent=2)+'\n');completed.append(metadata)
        print(json.dumps(dict(video=number+1,total=len(manifest),dataset=rec.dataset,view=view,frames=len(raw),elapsed_s=round(time.perf_counter()-started,1),gpu_mb=round(torch.cuda.max_memory_allocated()/2**20))),flush=True)
    suffix='' if args.workers==1 else f'_worker{args.worker_index}'
    (out/f'manifest{suffix}.json').write_text(json.dumps(completed,indent=2)+'\n');print('Visual extraction completed',flush=True)


if __name__=='__main__':main()
