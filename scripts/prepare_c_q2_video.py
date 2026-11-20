#!/usr/bin/env python3
"""Rebuild Q2 video features using only pixels and stateless image detections.

Reads no hand pose/quality/camera pose arrays. Official split metadata is checked
before decoding. CPU Farneback and image diagnostics; no GPU claim is made.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import pandas as pd
from shuwei_c.data import DEFAULT_DATA
from shuwei_c.q2_video import FEATURE_NAMES, ImageFeatures, VIEWS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(args):
    return dict(schema='pure_image_motion_36', feature_names=FEATURE_NAMES,
                width=args.width, sample_rate_hz=15., cv2=cv2.__version__, numpy=np.__version__,
                sources={str(p): digest(p) for p in [Path(__file__), Path('src/shuwei_c/q2_video.py'),
                                                     args.detections/'video_manifest.json']},
                detection_fields=['time_s', 'frame_index', 'width', 'height', 'detections.landmarks',
                                  'detections.handedness_score'],
                ignored_fields=['detections.label', 'landmark_z'],
                numeric_pose_or_quality_read=False,
                association='unordered image detections; previous/current image centres only',
                background='median normalized flow outside previous/current image-only detection boxes',
                available_time='source_time in nominal replay; add explicit processing_delay_s via provider',
                timing='CPU feature time excludes reused MediaPipe detection; decoding separately included in total')


def process(item, args, specification):
    target = args.output/f"{item['episode']}__{item['view']}.npz"
    meta_path = target.with_suffix('.json')
    detection_path = args.detections/item['features_file']
    source = args.data_root/item['source']
    expected = dict(identity=specification, detection_sha256=digest(detection_path),
                    source_size=source.stat().st_size, source_mtime_ns=source.stat().st_mtime_ns,
                    source_manifest=item)
    # JSON normalization avoids tuple/list mismatches on restart.
    expected = json.loads(json.dumps(expected))
    if target.exists() and meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing['inputs'] != expected:
            raise ValueError(f'Cache identity changed: {target.name}; use a new output directory')
        if digest(target) != existing['output_sha256']:
            raise ValueError(f'Cache hash changed: {target.name}')
        return existing
    rows = [json.loads(line) for line in detection_path.open()]
    if len(rows) != item['sampled_frames']:
        raise ValueError('Detection count disagrees with manifest')
    if any(b['frame_index'] <= a['frame_index'] or b['time_s'] <= a['time_s'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Non-increasing source frame/timestamp sequence')
    extractor = ImageFeatures(args.width)
    features, elapsed = [], []
    cap = cv2.VideoCapture(str(source))
    started = time.perf_counter(); index = 0; wanted = 0
    while wanted < len(rows):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode source frame {rows[wanted]["frame_index"]}')
        if index == rows[wanted]['frame_index']:
            row = rows[wanted]
            if frame.shape[:2] != (row['height'], row['width']):
                raise ValueError('Image dimensions disagree with detector input')
            began = time.perf_counter()
            features.append(extractor.step(frame, row['time_s'], row['detections']))
            elapsed.append(time.perf_counter()-began)
            wanted += 1
        index += 1
    cap.release()
    source_time = np.array([row['time_s'] for row in rows], float)
    features = np.asarray(features, np.float32)
    temporary = target.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, source_time=source_time, available_time=source_time,
                        frame_index=np.array([r['frame_index'] for r in rows]),
                        features=features, feature_names=np.array(FEATURE_NAMES),
                        feature_compute_s=np.array(elapsed))
    temporary.replace(target)
    meta = dict(inputs=expected, output_sha256=digest(target), frames=len(rows),
                elapsed_s=time.perf_counter()-started, feature_compute_total_s=float(sum(elapsed)),
                feature_compute_median_s=float(np.median(elapsed)),
                feature_compute_p95_s=float(np.quantile(elapsed, .95)),
                has_detection_frames=int(np.count_nonzero(features[:, FEATURE_NAMES.index('detection_count')])),
                has_matched_motion_frames=int(np.count_nonzero(features[:, FEATURE_NAMES.index('matched_count')])),
                split=item['split'], dataset=item['dataset'], episode=item['episode'], view=item['view'])
    meta_path.write_text(json.dumps(meta, indent=2)+'\n')
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--detections', type=Path, default=Path('artifacts/c_video_v2_15hz'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/c_q2/video/image_motion'))
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--limit-videos', type=int)
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()
    if args.width <= 0 or args.workers <= 0 or not 0 <= args.worker_index < args.workers:
        raise ValueError('Invalid worker/width options')
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    manifest = json.loads((args.detections/'video_manifest.json').read_text())
    split = pd.read_csv(args.data_root/'Restocking_Tasks_and_Constraints/record_split.csv').set_index('episode_uuid')
    for item in manifest:
        if item['split'] not in ('train', 'validation') or split.loc[item['episode'], 'split'] != item['split']:
            raise ValueError('Video manifest is not restricted to official development records')
        if item['mode'] != 'IMAGE' or item['view'] not in VIEWS:
            raise ValueError('Unverified detector mode/view')
    if args.limit_videos is not None:
        manifest = manifest[:args.limit_videos]
    specification = identity(args)
    rows = []
    for i, item in enumerate(manifest):
        if not args.merge_only and i % args.workers != args.worker_index:
            continue
        if args.merge_only:
            meta = json.loads((args.output/f"{item['episode']}__{item['view']}.json").read_text())
            if meta['inputs']['identity'] != json.loads(json.dumps(specification)):
                raise ValueError('Worker cache schema mismatch')
        else:
            meta = process(item, args, specification)
        rows.append(meta)
        print(json.dumps(dict(video=i+1, worker=args.worker_index, view=item['view'],
                              split=item['split'], frames=meta['frames'], elapsed_s=meta['elapsed_s'])), flush=True)
    suffix = '' if args.merge_only or args.workers == 1 else f'_worker{args.worker_index}'
    result = dict(identity=specification, test_used=False, videos=len(rows), frames=sum(r['frames'] for r in rows),
                  records=len(set(r['episode'] for r in rows)), metadata=rows,
                  complete_development_set=(len(rows) == 166 and sum(r['frames'] for r in rows) == 74100),
                  gpu_used=False, numeric_pose_or_quality_read=False)
    (args.output/f'manifest{suffix}.json').write_text(json.dumps(result, indent=2)+'\n')
    pd.DataFrame([{k: r[k] for k in ['split', 'dataset', 'episode', 'view', 'frames', 'elapsed_s',
                                   'feature_compute_total_s', 'feature_compute_median_s', 'feature_compute_p95_s',
                                   'has_detection_frames', 'has_matched_motion_frames']} for r in rows]).to_csv(
                                       args.output/f'summary{suffix}.csv', index=False)


if __name__ == '__main__':
    main()
