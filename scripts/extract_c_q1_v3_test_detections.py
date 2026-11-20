#!/usr/bin/env python3
"""Fixed 15 Hz stateless head-view detections for the explicitly observed test."""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import mediapipe as mp
from shuwei_c.data import Dataset
from prepare_c_video import videos


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker-index',type=int,default=0);parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
    root=Path('artifacts/c_q1_v3/frozen_quality');spec=json.loads((root/'freeze.json').read_text());out=root/'test_video/detections';out.mkdir(parents=True,exist_ok=True)
    for p,h in spec['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    weight=Path(spec['visual']['detector_weight']);assert mp.__version__==spec['visual']['mediapipe']
    options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(weight)),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=2)
    data=Dataset();items=[(r,v) for r in data.records('test') for v in videos(r) if v['view'] in ['head_left','head_right']];manifest=[];start=time.perf_counter();cv2.setNumThreads(2)
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for number,(rec,video) in enumerate(items):
            if number%args.workers!=args.worker_index:continue
            target=out/f'{rec.episode}__{video["view"]}.jsonl';meta=target.with_suffix('.metadata.json')
            if target.exists() and meta.exists():manifest.append(json.loads(meta.read_text()));continue
            cap=cv2.VideoCapture(str(video['path']));next_sample=0.;frame_index=0;rows=[]
            while True:
                ok,frame=cap.read()
                if not ok:break
                stamp=float(video['times'][frame_index])
                if stamp+1e-9>=next_sample:
                    next_sample+=1/15;height,width=frame.shape[:2]
                    rgb=cv2.cvtColor(cv2.resize(frame,(960,round(height*960/width))),cv2.COLOR_BGR2RGB)
                    result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)));detections=[]
                    for hand,handedness in zip(result.hand_landmarks,result.handedness):detections.append(dict(label=handedness[0].category_name,handedness_score=float(handedness[0].score),landmarks=[[float(p.x),float(p.y),float(p.z)] for p in hand]))
                    rows.append(dict(time_s=stamp,frame_index=frame_index,width=width,height=height,detections=detections))
                frame_index+=1
            cap.release();assert frame_index==video['count'];temp=target.with_suffix('.tmp');temp.write_text(''.join(json.dumps(row)+'\n' for row in rows));temp.replace(target)
            identity=dict(episode=rec.episode,dataset=rec.dataset,environment=rec.environment,split='test',view=video['view'],source=str(video['path'].relative_to(data.root)),fps=video['fps'],frame_count=video['count'],time_basis=video['time_basis'],sample_rate_hz=15.,model_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),mode='IMAGE',mediapipe=mp.__version__,features_file=target.name,sampled_frames=len(rows))
            meta.write_text(json.dumps(identity,indent=2)+'\n');manifest.append(identity);print(json.dumps(dict(video=number+1,total=len(items),frames=len(rows),elapsed_s=time.perf_counter()-start)),flush=True)
    (out/f'video_manifest_worker{args.worker_index}.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
