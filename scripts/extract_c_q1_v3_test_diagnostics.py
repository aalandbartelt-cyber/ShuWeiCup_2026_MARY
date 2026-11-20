#!/usr/bin/env python3
"""Same development crops/flow/assignment diagnostics; frozen test head views."""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from shuwei_c.data import Dataset
from extract_c_q1_v3_visual import projected_hands,crop_box,region_stats


def main():
    root=Path('artifacts/c_q1_v3/frozen_quality');spec=json.loads((root/'freeze.json').read_text());source=root/'test_video/detections';out=root/'test_video/diagnostics';out.mkdir(parents=True,exist_ok=True)
    for p,h in spec['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    items=json.loads((source/'video_manifest.json').read_text());assert len(items)==14
    data=Dataset();manifest=[];start=time.perf_counter();cv2.setNumThreads(2)
    for item in items:
        rec=data.record(item['episode']);view=item['view'];target=out/f'{rec.episode}__{view}.npz';meta=target.with_suffix('.metadata.json')
        if target.exists() and meta.exists():manifest.append(json.loads(meta.read_text()));continue
        rows=[json.loads(s) for s in (source/item['features_file']).read_text().splitlines()];vt=np.array([r['time_s'] for r in rows]);projected,visible=projected_hands(rec,view,vt)
        cap=cv2.VideoCapture(str(data.root/item['source']));wanted={r['frame_index']:k for k,r in enumerate(rows)};frame_index=0;previous=None;previous_t=None;diagnostics=[];available=[]
        while True:
            ok,image=cap.read()
            if not ok:break
            if frame_index not in wanted:frame_index+=1;continue
            index=wanted[frame_index];row=rows[index];height,width=image.shape[:2];detected=[np.array(d['landmarks'])[:,:2]*[width,height] for d in row['detections']]
            boxes=[None,None];slots=np.zeros((2,4),np.float32);expected=projected[index].mean(axis=1)
            for hand in range(2):
                if visible[index,hand]:boxes[hand]=crop_box(projected[index,hand],width,height)
            if detected:
                centers=np.array([d.mean(axis=0) for d in detected]);distance=np.linalg.norm((expected[:,None]-centers[None,:,:])/[width,height],axis=-1);ah,ad=linear_sum_assignment(distance)
                for hand,detection in zip(ah,ad):
                    slots[hand]=[1,float(row['detections'][detection]['handedness_score']),min(distance[hand,detection],2.),len(detected)]
                    if distance[hand,detection]<.2:boxes[hand]=crop_box(detected[detection],width,height)
            all_boxes=[np.array([0,0,width,height]),*boxes];small=cv2.resize(image,(320,round(height*320/width)),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
            dt=0. if previous is None else float(vt[index]-previous_t);flow=np.zeros((*gray.shape,2),np.float32) if previous is None else cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,15,3,5,1.2,0)
            background=np.median(flow.reshape(-1,2),axis=0);ratio=np.array([gray.shape[1]/width,gray.shape[0]/height]*2);stats=[]
            for box in all_boxes:stats.extend(region_stats(gray,flow,None if box is None else np.rint(box*ratio).astype(int),background,dt))
            diagnostics.append(np.r_[stats,slots.ravel()]);available.append([box is not None for box in all_boxes]);previous=gray;previous_t=vt[index];frame_index+=1
        cap.release();assert len(diagnostics)==len(rows)
        temp=target.with_suffix('.tmp.npz');np.savez_compressed(temp,time_s=vt,diagnostics=np.array(diagnostics),available=np.array(available));temp.replace(target)
        identity=dict(episode=rec.episode,dataset=rec.dataset,split='test',view=view,frames=len(rows),source_detections_sha256=hashlib.sha256((source/item['features_file']).read_bytes()).hexdigest())
        meta.write_text(json.dumps(identity,indent=2)+'\n');manifest.append(identity);print(json.dumps(dict(video=len(manifest),frames=len(rows),elapsed_s=time.perf_counter()-start)),flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
