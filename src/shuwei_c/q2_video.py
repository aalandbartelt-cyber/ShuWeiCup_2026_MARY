"""Image-only Q2 features with explicit source and availability times.

No recording-world coordinates, numerical quality labels, camera poses, source
identities or future frames enter the extractor/provider outputs. MediaPipe's
handedness score is a detector output, never a measured accuracy probability.
AI-assisted implementation: OpenAI Codex, 2026-11-20.
"""
from pathlib import Path
from types import MappingProxyType
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


VIEWS = ('head_left', 'head_right', 'left_wrist', 'right_wrist')
FEATURE_NAMES = (
    'gray_mean', 'gray_std', 'log_sharpness', 'clipped_fraction',
    'flow_known', 'background_flow_x_per_s', 'background_flow_y_per_s',
    'global_residual_median_per_s', 'global_residual_p90_per_s',
    'detection_count', 'detector_handedness_score_mean',
    'detection_area_mean', 'detection_area_max',
    'detection_center_x_mean', 'detection_center_y_mean',
    'detection_center_x_std', 'detection_center_y_std',
    'detection_boundary_min', 'detection_boundary_mean',
    'detection_spread_mean', 'detection_union_fraction',
    'hand_region_known', 'hand_log_sharpness', 'hand_clipped_fraction',
    'hand_residual_median_per_s', 'hand_residual_p90_per_s',
    'hand_residual_x_per_s', 'hand_residual_y_per_s',
    'matched_count', 'matched_fraction', 'center_motion_median_per_s',
    'center_motion_max_per_s', 'center_residual_motion_median_per_s',
    'center_velocity_x_per_s', 'center_velocity_y_per_s', 'frame_gap_s',
)


def immutable(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def image_detections(detections):
    """Convert normalized image landmarks; ignore anatomical labels entirely."""
    out = []
    for detection in detections:
        points = np.asarray(detection['landmarks'], dtype=float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            continue
        xy = points[:, :2]
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        centre = xy.mean(axis=0)
        if not ((hi > 0).all() and (lo < 1).all()):
            continue
        lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
        area = float(np.prod(np.maximum(hi-lo, 0)))
        if area <= 0:
            continue
        out.append(dict(centre=centre, lo=lo, hi=hi, area=area,
                        boundary=float(min(*lo, *(1-hi))),
                        spread=float(np.sqrt(np.mean(np.sum((xy-centre)**2, axis=1)))),
                        handedness_score=float(detection.get('handedness_score', 0))))
    # Stable order makes ambiguous Hungarian ties independent of detector order.
    return sorted(out, key=lambda r: tuple(r['centre'])+tuple(r['lo']))


def detection_mask(detections, height, width):
    mask = np.zeros((height, width), dtype=bool)
    for det in detections:
        lo, hi = det['lo'], det['hi']
        # Image-only modest padding; no 3D projection or future association.
        pad = np.maximum((hi-lo)*.1, .005)
        x0, y0 = np.floor(np.maximum(0, lo-pad)*[width, height]).astype(int)
        x1, y1 = np.ceil(np.minimum(1, hi+pad)*[width, height]).astype(int)
        mask[y0:y1, x0:x1] = True
    return mask


class ImageFeatures:
    """Sequential two-frame optical flow and unordered detection summaries."""
    def __init__(self, width=320):
        self.width = int(width)
        self.previous_gray = None
        self.previous_time = None
        self.previous_detections = []

    def step(self, image, source_time, detections):
        import cv2
        source_time = float(source_time)
        if not np.isfinite(source_time):
            raise ValueError('Nonfinite image timestamp')
        if self.previous_time is not None and source_time <= self.previous_time:
            raise ValueError('Frames must arrive in strictly increasing source time')
        height, width = image.shape[:2]
        small = cv2.resize(image, (self.width, max(1, round(height*self.width/width))),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        current = image_detections(detections)
        mask = detection_mask(current, h, w)
        dt = source_time-self.previous_time if self.previous_time is not None else 0.
        flow_known = self.previous_gray is not None and self.previous_gray.shape == gray.shape and dt > 0
        flow = np.zeros((h, w, 2), np.float32)
        if flow_known:
            flow = cv2.calcOpticalFlowFarneback(self.previous_gray, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            flow = flow/np.array([w, h], np.float32)/dt
        previous_mask = detection_mask(self.previous_detections, h, w)
        background_mask = ~(mask | previous_mask)
        if background_mask.mean() < .1:
            background_mask = np.ones((h, w), bool)
        background = np.median(flow[background_mask], axis=0)
        residual = flow-background
        magnitude = np.linalg.norm(residual, axis=-1)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        clipped = (gray < 8) | (gray > 247)
        values = {name: 0. for name in FEATURE_NAMES}
        values.update(gray_mean=float(gray.mean()/255), gray_std=float(gray.std()/255),
                      log_sharpness=float(np.log1p(laplacian.var())),
                      clipped_fraction=float(clipped.mean()), flow_known=float(flow_known),
                      background_flow_x_per_s=float(background[0]),
                      background_flow_y_per_s=float(background[1]),
                      global_residual_median_per_s=float(np.median(magnitude)),
                      global_residual_p90_per_s=float(np.quantile(magnitude, .9)),
                      detection_count=float(len(current)), detection_union_fraction=float(mask.mean()),
                      frame_gap_s=float(dt))
        if current:
            centres = np.stack([d['centre'] for d in current])
            areas = np.array([d['area'] for d in current])
            boundary = np.array([d['boundary'] for d in current])
            values.update(detector_handedness_score_mean=float(np.mean([d['handedness_score'] for d in current])),
                          detection_area_mean=float(areas.mean()), detection_area_max=float(areas.max()),
                          detection_center_x_mean=float(centres[:, 0].mean()),
                          detection_center_y_mean=float(centres[:, 1].mean()),
                          detection_center_x_std=float(centres[:, 0].std()),
                          detection_center_y_std=float(centres[:, 1].std()),
                          detection_boundary_min=float(boundary.min()),
                          detection_boundary_mean=float(boundary.mean()),
                          detection_spread_mean=float(np.mean([d['spread'] for d in current])))
        if mask.any():
            values.update(hand_region_known=1., hand_log_sharpness=float(np.log1p(laplacian[mask].var())),
                          hand_clipped_fraction=float(clipped[mask].mean()),
                          hand_residual_median_per_s=float(np.median(magnitude[mask])),
                          hand_residual_p90_per_s=float(np.quantile(magnitude[mask], .9)),
                          hand_residual_x_per_s=float(np.median(residual[mask, 0])),
                          hand_residual_y_per_s=float(np.median(residual[mask, 1])))
        if current and self.previous_detections and flow_known:
            centres = np.stack([d['centre'] for d in current])
            previous = np.stack([d['centre'] for d in self.previous_detections])
            distance = np.linalg.norm(centres[:, None]-previous[None], axis=-1)
            a, b = linear_sum_assignment(distance)
            keep = distance[a, b] <= .25
            a, b = a[keep], b[keep]
            if len(a):
                velocity = (centres[a]-previous[b])/dt
                speed = np.linalg.norm(velocity, axis=-1)
                residual_speed = np.linalg.norm(velocity-background, axis=-1)
                values.update(matched_count=float(len(a)), matched_fraction=float(len(a)/max(len(current), len(previous))),
                              center_motion_median_per_s=float(np.median(speed)),
                              center_motion_max_per_s=float(speed.max()),
                              center_residual_motion_median_per_s=float(np.median(residual_speed)),
                              center_velocity_x_per_s=float(np.median(velocity[:, 0])),
                              center_velocity_y_per_s=float(np.median(velocity[:, 1])))
        features = np.array([values[name] for name in FEATURE_NAMES], np.float32)
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite image-derived features')
        self.previous_gray = gray.copy()
        self.previous_time = source_time
        self.previous_detections = current
        return features


class VideoProvider:
    """As-of image snapshots, optionally delayed or restricted to view groups.

    Each view contributes 36 image features plus available/source-age/delay.
    Missing views have an all-zero block and available=0. `at()` contains no
    file paths or recording identifiers; timestamp traces remain inspectable.
    """
    def __init__(self, streams, processing_delay_s=0., views=VIEWS):
        if processing_delay_s < 0 or not np.isfinite(processing_delay_s):
            raise ValueError('Processing delay must be finite and nonnegative')
        self.views = tuple(views)
        if not set(self.views) <= set(VIEWS):
            raise ValueError('Unknown view')
        self.streams = {}
        for view, stream in streams.items():
            if view not in self.views:
                continue
            source = np.asarray(stream['source_time'], float)
            available = np.asarray(stream.get('available_time', source), float)+processing_delay_s
            features = np.asarray(stream['features'], np.float32)
            if (source.ndim != 1 or available.shape != source.shape or
                features.shape != (len(source), len(FEATURE_NAMES)) or
                not np.isfinite(source).all() or not np.isfinite(available).all() or
                not np.isfinite(features).all() or np.any(np.diff(source) <= 0) or
                np.any(np.diff(available) < 0) or np.any(available < source)):
                raise ValueError('Invalid causal video cache')
            self.streams[view] = dict(source_time=immutable(source),
                                      available_time=immutable(available), features=immutable(features))
        self.feature_names = tuple(f'{view}_{name}' for view in self.views
                                   for name in ('available', 'frame_age_s', 'processing_delay_s')+FEATURE_NAMES)

    @classmethod
    def from_cache(cls, root, episode, processing_delay_s=0., views=VIEWS):
        streams = {}
        root = Path(root)
        for view in views:
            path = root/f'{episode}__{view}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    if tuple(data['feature_names'].tolist()) != FEATURE_NAMES:
                        raise ValueError(f'Wrong video feature schema: {view}')
                    streams[view] = {name: data[name].copy() for name in ('source_time', 'available_time', 'features')}
        return cls(streams, processing_delay_s=processing_delay_s, views=views)

    def at(self, now):
        if not np.isfinite(now):
            raise ValueError('Invalid query time')
        blocks, trace = [], {}
        for view in self.views:
            block = np.zeros(3+len(FEATURE_NAMES), np.float32)
            if view in self.streams:
                stream = self.streams[view]
                i = np.searchsorted(stream['available_time'], now, side='right')-1
                if i >= 0:
                    source = float(stream['source_time'][i]); available = float(stream['available_time'][i])
                    block[:3] = [1., now-source, available-source]
                    block[3:] = stream['features'][i]
                    trace[view] = MappingProxyType(dict(source_time=source, available_time=available,
                                                       frame_age_s=float(now-source)))
            blocks.append(block)
        return MappingProxyType(dict(features=immutable(np.concatenate(blocks)),
                                     feature_names=self.feature_names,
                                     views=MappingProxyType(trace)))

    def vector(self, now, lags=(0.,)):
        if any(lag < 0 for lag in lags):
            raise ValueError('Lags must be nonnegative')
        return immutable(np.concatenate([self.at(now-lag)['features'] for lag in lags]))
