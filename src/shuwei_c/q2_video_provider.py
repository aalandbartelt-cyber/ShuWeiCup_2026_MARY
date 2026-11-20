"""Verified complete-cache loaders for independent Q2 image/tracker branches.

This adapter is separate from the running extraction sources. It distinguishes
physically absent views from incomplete extraction, and can replay shared GPU
service rather than implicitly assuming one independent GPU per camera.
"""
from pathlib import Path
from types import MappingProxyType
import hashlib
import json
import numpy as np
from .q2_video import VIEWS, FEATURE_NAMES, immutable

TRACK_FEATURE_NAMES = (
    'visible_fraction', 'visible_pair_fraction', 'median_dx_per_s', 'median_dy_per_s',
    'motion_median_per_s', 'motion_p90_per_s', 'residual_motion_median_per_s',
    'residual_motion_p90_per_s', 'hand_tracks_fraction', 'hand_motion_median_per_s',
    'hand_residual_motion_median_per_s', 'hand_tracks_known',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AsOfVideoFeatures:
    def __init__(self, streams, names, views=VIEWS, processing_delay_s=0.):
        if not np.isfinite(processing_delay_s) or processing_delay_s < 0:
            raise ValueError('Invalid additional processing delay')
        if not set(views) <= set(VIEWS):
            raise ValueError('Unknown video view')
        self.views = tuple(views); self.names = tuple(names)
        self.streams = {}
        for view in self.views:
            if view not in streams:
                continue
            source = np.asarray(streams[view]['source_time'], float)
            arrival = np.asarray(streams[view]['available_time'], float)+processing_delay_s
            values = np.asarray(streams[view]['features'], np.float32)
            if (source.ndim != 1 or arrival.shape != source.shape or
                values.shape != (len(source), len(names)) or
                not np.isfinite(source).all() or not np.isfinite(arrival).all() or
                not np.isfinite(values).all() or np.any(np.diff(source) <= 0) or
                np.any(np.diff(arrival) < 0) or np.any(arrival < source)):
                raise ValueError('Invalid source/availability times or feature schema')
            self.streams[view] = dict(source_time=immutable(source), available_time=immutable(arrival),
                                      features=immutable(values))
        self.feature_names = tuple(f'{view}_{name}' for view in self.views
                                   for name in ('available','frame_age_s','processing_delay_s')+self.names)

    def at(self, now):
        if not np.isfinite(now):
            raise ValueError('Invalid query time')
        blocks=[]; trace={}
        for view in self.views:
            block=np.zeros(3+len(self.names),np.float32)
            if view in self.streams:
                row=self.streams[view]
                index=np.searchsorted(row['available_time'],now,side='right')-1
                if index>=0:
                    source=float(row['source_time'][index]); arrival=float(row['available_time'][index])
                    block[:3]=[1.,now-source,arrival-source]
                    block[3:]=row['features'][index]
                    trace[view]=MappingProxyType(dict(source_time=source,available_time=arrival,frame_age_s=now-source))
            blocks.append(block)
        return MappingProxyType(dict(features=immutable(np.concatenate(blocks)),feature_names=self.feature_names,
                                     views=MappingProxyType(trace)))

    def vector(self, now, lags=(0.,)):
        if any(lag<0 for lag in lags):
            raise ValueError('Negative history lag')
        return immutable(np.concatenate([self.at(now-lag)['features'] for lag in lags]))


class VerifiedVideoCache:
    """A complete development cache and its physical-view inventory.

    Construct once per experiment, then `for_episode` returns a prefix-only
    provider. This metadata/index object is never passed to the online policy.

    time_mode:
      nominal: source/chunk-end availability (no processing latency claim);
      measured: independently queued measured feature/GPU service per stream;
      serial_gpu: all selected tracker views share one serial GPU queue.

    Measured GPU service excludes the original MediaPipe detection/decoding
    pipeline. These are explicit latency scenarios, not physical robot trials.
    """
    def __init__(self, root, kind='image', source_manifest='artifacts/c_video_v2_15hz/video_manifest.json'):
        self.root=Path(root); self.kind=kind
        if kind not in ('image','tracker'):
            raise ValueError('Expected image or tracker cache')
        path=self.root/'manifest.json'
        if not path.exists():
            raise ValueError('Video cache incomplete: final manifest is absent')
        self.manifest=json.loads(path.read_text())
        if self.manifest.get('complete_development_set') is not True:
            raise ValueError('Video cache is not a completed development extraction')
        source=json.loads(Path(source_manifest).read_text())
        if any(row['split'] not in ('train','validation') for row in source):
            raise ValueError('Development loader refuses test records')
        expected={(row['episode'],row['view']):row for row in source}
        actual={(row['episode'],row['view']):row for row in self.manifest['metadata']}
        if len(expected)!=len(source) or len(actual)!=len(self.manifest['metadata']) or set(actual)!=set(expected):
            raise ValueError('Physical-view manifest and completed cache inventory disagree')
        for key,row in actual.items():
            if row['inputs']['source_manifest']!=expected[key] or row['frames']!=expected[key]['sampled_frames']:
                raise ValueError('Cache split/view/frame accounting disagrees with source')
        self.index=actual; self.source=expected
        self.names=FEATURE_NAMES if kind=='image' else TRACK_FEATURE_NAMES

    def for_episode(self, episode, views=VIEWS, time_mode='nominal', processing_delay_s=0.):
        if time_mode not in ('nominal','measured','serial_gpu'):
            raise ValueError('Unknown availability scenario')
        if time_mode=='serial_gpu' and self.kind!='tracker':
            raise ValueError('Serial GPU mode requires the GPU tracker cache')
        physical={view for ep,view in self.index if ep==episode}
        if not physical:
            raise KeyError('Episode is absent from the completed development cache')
        streams={}
        for view in views:
            if view not in physical:
                continue  # genuinely absent physical stream
            path=self.root/f'{episode}__{view}.npz'
            row=self.index[(episode,view)]
            if not path.exists():
                raise ValueError(f'Expected physical view has no cache: {view}')
            if digest(path)!=row['output_sha256']:
                raise ValueError(f'Video cache output hash mismatch: {view}')
            with np.load(path,allow_pickle=False) as data:
                if tuple(data['feature_names'].tolist())!=self.names:
                    raise ValueError('Video feature names differ from expected schema')
                source=data['source_time'].copy(); values=data['features'].copy()
                if self.kind=='image':
                    expected_rows=row['frames']; service=data['feature_compute_s'].copy()
                else:
                    expected_rows=max(0,(row['frames']-16)//8+1)
                    service=data['inference_s'].copy()
                if len(source)!=expected_rows or service.shape!=source.shape or np.any(service<0) or not np.isfinite(service).all():
                    raise ValueError('Video cache row/service-time accounting mismatch')
                arrival=source.copy()
                if time_mode=='measured':
                    previous=-np.inf
                    for i,stamp in enumerate(source):
                        previous=max(stamp,previous)+service[i]; arrival[i]=previous
                streams[view]=dict(source_time=source,available_time=arrival,features=values,service=service)
        if time_mode=='serial_gpu':
            events=sorted((float(stream['source_time'][i]),view,i,float(service))
                          for view,stream in streams.items() for i,service in enumerate(stream['service']))
            previous=-np.inf
            for source,view,i,service in events:
                previous=max(source,previous)+service
                streams[view]['available_time'][i]=previous
        return AsOfVideoFeatures(streams,self.names,views,processing_delay_s)
