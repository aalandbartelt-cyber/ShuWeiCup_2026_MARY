#!/usr/bin/env python3
"""All-view, timestamped visual baseline and blinded human annotation packet.

AI: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
IMAGE mode is stateless: no hidden intervening frames enter Q2 replay features.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset,DEFAULT_DATA

MODEL_URL='https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'


def videos(rec):
    result=[]
    for path in sorted(rec.folder.rglob('*.mp4')):
        view=path.parents[1].name.replace('observation.images.','') if rec.dataset=='EgoDemo' else path.stem
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():raise RuntimeError(f'Cannot decode {path.name}')
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if rec.dataset=='EgoDemo':
            meta=pd.read_parquet(rec.folder/'meta/episodes/chunk-000/file-000.parquet')
            first=float(meta[f'videos/observation.images.{view}/from_timestamp'].iloc[0])
            times=np.arange(count)/fps-first
            basis='video frame index / container FPS minus episode from_timestamp; approximate alignment'
        else:
            frames=pd.read_csv(path.with_name(path.stem+'_frames.csv'))
            if len(frames)!=count:raise ValueError(f'Video/frame-table count mismatch: {view}')
            times=frames['_offset_from_first_log_s'].to_numpy(float)
            basis='recorded frame-table relative timestamps'
        result.append(dict(path=path,view=view,times=times,fps=fps,count=count,time_basis=basis))
    return result


def process(data,out,rate,limit=None,worker_index=None,workers=1):
    import mediapipe as mp
    model=out/'hand_landmarker.task'
    if not model.exists():urllib.request.urlretrieve(MODEL_URL,model)
    options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=2)
    manifest=[];index=0
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for split in ['train','validation']:
            for rec in data.records(split):
                for video in videos(rec):
                    if limit is not None and index>=limit:break
                    index+=1
                    if worker_index is not None and (index-1)%workers!=worker_index:continue
                    target=out/f'{rec.episode}__{video["view"]}.jsonl'
                    identity=dict(episode=rec.episode,dataset=rec.dataset,environment=rec.environment,split=rec.split,
                        view=video['view'],source=str(video['path'].relative_to(data.root)),fps=video['fps'],
                        frame_count=video['count'],time_basis=video['time_basis'],sample_rate_hz=rate,
                        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),mode='IMAGE',mediapipe=mp.__version__)
                    if not target.exists():
                        cap=cv2.VideoCapture(str(video['path']));next_sample=0.;frame_index=0;rows=[]
                        while True:
                            ok,frame=cap.read()
                            if not ok:break
                            stamp=float(video['times'][frame_index])
                            if stamp+1e-9>=next_sample:
                                next_sample+=1/rate
                                # Preserve aspect ratio; normalized 2D output independent of resize.
                                height,width=frame.shape[:2]
                                rgb=cv2.cvtColor(cv2.resize(frame,(960,round(height*960/width))),cv2.COLOR_BGR2RGB)
                                result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)))
                                detections=[]
                                for hand,handedness in zip(result.hand_landmarks,result.handedness):
                                    detections.append(dict(label=handedness[0].category_name,
                                        handedness_score=float(handedness[0].score),
                                        landmarks=[[float(p.x),float(p.y),float(p.z)] for p in hand]))
                                rows.append(dict(time_s=stamp,frame_index=frame_index,width=width,height=height,detections=detections))
                            frame_index+=1
                        cap.release()
                        if frame_index!=video['count']:raise RuntimeError('Incomplete video decoding')
                        temp=target.with_suffix('.tmp');temp.write_text(''.join(json.dumps(row)+'\n' for row in rows));temp.replace(target)
                    identity['features_file']=target.name
                    identity['sampled_frames']=sum(1 for _ in target.open())
                    manifest.append(identity)
                    print(json.dumps(dict(video=index,dataset=rec.dataset,view=video['view'],split=split,samples=identity['sampled_frames'])),flush=True)
    suffix='' if worker_index is None else f'_worker{worker_index}'
    (out/f'video_manifest{suffix}.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    (out/f'model_provenance{suffix}.json').write_text(json.dumps(dict(url=MODEL_URL,sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        mediapipe=mp.__version__,coordinate_warning='World landmarks are hand-centred, not recording-world translations.',
        handedness_warning='Unverified mirror convention; scores are not accuracy confidence.',test_used=False),indent=2)+'\n')


def annotation_packet(data,out):
    packet=out/'human_2d';packet.mkdir(parents=True,exist_ok=True)
    candidates={}
    for split in ['train','validation']:
        for rec in data.records(split):
            t,p,bad=rec.hands()
            for video in videos(rec):
                for quality in [False,True]:
                    ix=np.flatnonzero(bad.any(axis=1)==quality)
                    if not len(ix):continue
                    key=(split,rec.dataset,video['view'],quality)
                    # One central candidate per stratum; IDs stay local.
                    if key not in candidates:
                        target=t[ix[len(ix)//2]]
                        frame=int(np.argmin(np.abs(video['times']-target)))
                        candidates[key]=(rec,video,frame,float(video['times'][frame]))
    labels=[];internal=[]
    rng=np.random.default_rng(20261120);items=list(candidates.items());rng.shuffle(items)
    for i,(stratum,(rec,video,frame,stamp)) in enumerate(items):
        name=f'frame_{i+1:03d}.jpg';cap=cv2.VideoCapture(str(video['path']));cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,image=cap.read();cap.release()
        if not ok:raise RuntimeError('Annotation frame decode failed')
        cv2.imwrite(str(packet/name),image)
        # Blind human to original flags and automatic predictions.
        labels.append(dict(image=name,left_visible='',right_visible='',left_x='',left_y='',right_x='',right_y='',
                           reviewer='',notes=''))
        internal.append(dict(image=name,split=rec.split,dataset=rec.dataset,episode=rec.episode,view=video['view'],
                             flagged=bool(stratum[-1]),frame_index=frame,time_s=stamp))
    with (packet/'annotations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(labels[0]));writer.writeheader();writer.writerows(labels)
    (out/'human_2d_sampling_audit.json').write_text(json.dumps(internal,indent=2)+'\n')
    (packet/'README.md').write_text('''# 独立二维人工核验

请先独立观察原图，不看自动关键点和质量标记。按画面中人的解剖左右手填写（不是画面左右）。
left_visible/right_visible 填 full / partial / absent / uncertain；可见手填写掌部中心像素 x,y。
无法辨认时留空坐标并写原因；不要猜遮挡部位。填写 reviewer 和 notes 后返回 annotations.csv。
坐标原点为图像左上角；x 向右，y 向下。请保留原图分辨率，查看器缩放坐标须换算。
该核验检验可见性、左右手与二维定位，仅可作为二维外部证据，不能证明三维世界坐标准确。
当前分层取样用于诊断，包含质量标记富集；总体发生率不能由此直接估计。
''')
    print(json.dumps(dict(annotation_frames=len(labels),packet=str(packet))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    p.add_argument('--output',type=Path,default=Path('artifacts/c_video'))
    p.add_argument('--rate',type=float,default=5.)
    p.add_argument('--limit-videos',type=int)
    p.add_argument('--annotations-only',action='store_true')
    p.add_argument('--worker-index',type=int)
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);data=Dataset(a.data_root)
    if a.annotations_only:annotation_packet(data,a.output)
    else:
        if a.worker_index is not None and not 0<=a.worker_index<a.workers:raise ValueError('Invalid worker partition')
        process(data,a.output,a.rate,a.limit_videos,a.worker_index,a.workers)


if __name__=='__main__':main()
