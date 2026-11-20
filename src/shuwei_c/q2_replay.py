"""Q2 event replay: private reference storage, explicit arrivals, immutable disclosure.

The predictor/policy interface has no record, path, reference, or future-quality
handle. Artificial-window selection is an evaluator operation, never a feature.
AI-assisted implementation: OpenAI Codex, 2026-11-20; verification in tests/.
"""
from dataclasses import dataclass, field
from functools import cached_property
from typing import Mapping
import hashlib
import heapq
import math
from types import MappingProxyType

import numpy as np

EPS = 1e-9


def readonly(value):
    a = np.array(value, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class History:
    time: np.ndarray
    points: np.ndarray
    bad: np.ndarray
    auxiliary: Mapping = field(default_factory=dict)
    revision: int = 0

    @cached_property
    def centres(self):
        return readonly(self.points.mean(axis=2))


@dataclass(frozen=True)
class Pending:
    request_id: int
    sample_time: float
    arrival_time: float


@dataclass(frozen=True)
class Context:
    now: float
    origin: float
    end: float
    last_score_time: float
    delay: float
    remaining: int
    pending: tuple
    history: History
    dataset: str
    video: object = None


@dataclass(frozen=True)
class Window:
    window_id: str
    episode: str
    dataset: str
    environment: str
    split: str
    origin_index: int
    stop: int
    origin: float
    horizon: float

    @property
    def end(self):
        return self.origin + self.horizon


def select_windows(record, config):
    """Select fully qualified target intervals; audit every attempted grid cell."""
    t, points, bad = record.hands()
    windows, audit = [], []
    horizons = config['main_windows_s'] + config['stress_windows_s']
    for nominal in np.arange(t[0] + config['warmup_s'], t[-1] + EPS,
                             config['window_stride_s']):
        origin_index = int(np.searchsorted(t, nominal + EPS, side='right') - 1)
        for horizon in horizons:
            origin = float(t[origin_index])
            stop = int(np.searchsorted(t, origin + horizon + EPS, side='right'))
            target = slice(origin_index + 1, stop)
            if origin_index < 2:
                reason = 'insufficient_prefix'
            elif origin + horizon > t[-1] + EPS:
                reason = 'record_end'
            elif stop <= origin_index + 1:
                reason = 'no_target_samples'
            elif not np.isfinite(points[target]).all():
                reason = 'nonfinite_reference'
            elif bad[target].any():
                reason = 'flagged_reference'
            elif not np.isfinite(points[:origin_index + 1]).all():
                reason = 'nonfinite_prefix_requires_explicit_handling'
            else:
                reason = 'included'
            key = f'{record.episode}:{origin_index}:{horizon:.9g}'
            window_id = hashlib.sha256(key.encode()).hexdigest()[:20]
            row = dict(window_id=window_id, episode=record.episode,
                       environment=record.environment, dataset=record.dataset,
                       split=record.split, nominal_start_s=float(nominal),
                       origin_index=origin_index, origin_s=origin,
                       horizon_s=float(horizon), stop=stop,
                       target_times=stop-origin_index-1, reason=reason)
            audit.append(row)
            if reason == 'included':
                windows.append(Window(window_id, record.episode, record.dataset,
                                      record.environment, record.split,
                                      origin_index, stop, origin, float(horizon)))
    return windows, audit


def _auxiliary_prefix(record, until, latest=False):
    result = {}
    for name in ('head', 'head_left_camera', 'head_right_camera'):
        if name not in record.streams:
            continue
        stream = record.streams[name]
        stop = np.searchsorted(stream['time'], until + EPS, side='right')
        if not stop:
            continue
        take = slice(stop-1 if latest else 0, stop)
        result[name] = {key: readonly(value[take]) for key, value in stream.items()}
    return result


def _freeze_auxiliary(auxiliary):
    return MappingProxyType({k: MappingProxyType(v) for k, v in auxiliary.items()})


def initial_history(record, window, numerical=None):
    t, points, bad = numerical if numerical is not None else record.hands()
    n = window.origin_index + 1
    return History(readonly(t[:n]), readonly(points[:n]), readonly(bad[:n]),
                   _freeze_auxiliary(_auxiliary_prefix(record, window.origin)))


def append_packet(history, sample_time, points, bad, auxiliary):
    if sample_time <= history.time[-1] + EPS:
        raise ValueError('Non-increasing/duplicate source time in disclosed packet')
    merged = {k: dict(v) for k, v in history.auxiliary.items()}
    for name, values in auxiliary.items():
        if name not in merged:
            merged[name] = values
        elif values['time'][0] > merged[name]['time'][-1] + EPS:
            merged[name] = {k: readonly(np.concatenate([merged[name][k], v]))
                            for k, v in values.items()}
    return History(readonly(np.r_[history.time, sample_time]),
                   readonly(np.concatenate([history.points, points[None]])),
                   readonly(np.concatenate([history.bad, bad[None]])),
                   _freeze_auxiliary(merged), history.revision+1)


def waiting_union(requests, origin, end):
    intervals = sorted((max(origin, r['sample_time']), min(end, r['arrival_time']))
                       for r in requests if r['sample_time'] < end)
    total, right = 0., origin
    for left, stop in intervals:
        if stop > max(left, right):
            total += stop - max(left, right)
        right = max(right, stop)
    return float(total)


def replay(record, window, predictor, policy, cap, delay, cost=1.,
           multiple_pending=True, video_provider=None, observer=None, numerical=None):
    """Replay one fixed window. Reference evaluation occurs after action/prediction.

    `observer`, if supplied, receives only Context and prediction, for collecting
    causal features. Target arrays are joined separately after replay.
    predictor(history, now) -> (2,3); policy(Context) -> bool.
    video_provider, if supplied, must expose an audited `at(now)` API.
    """
    if int(cap) != cap or cap < 0 or not math.isfinite(delay) or delay < 0:
        raise ValueError('Invalid request budget or delay')
    if not math.isfinite(cost) or cost < 0:
        raise ValueError('Invalid request cost')
    t, points, bad = numerical if numerical is not None else record.hands()
    indices = range(window.origin_index+1, window.stop)
    if not len(indices):
        raise ValueError('Empty window')
    history = initial_history(record, window, (t, points, bad))
    queue, requests, steps = [], [], []
    predictions = []

    def deliver(now, current, at_score=True):
        while queue and queue[0][0] <= now + EPS:
            arrival, source_index, request_id = heapq.heappop(queue)
            # Only the private replay environment accesses these references.
            current = append_packet(current, float(t[source_index]),
                                    points[source_index], bad[source_index],
                                    _auxiliary_prefix(record, t[source_index], latest=True))
            requests[request_id]['delivered'] = True
            requests[request_id]['delivered_at'] = float(arrival)
            requests[request_id]['first_usable_score_time'] = float(now) if at_score else None
        return current

    for i in indices:
        now = float(t[i])
        history = deliver(now, history)
        pending = tuple(Pending(rid, float(t[ix]), float(arrival))
                        for arrival, ix, rid in sorted(queue))
        context = Context(now, window.origin, window.end, float(t[window.stop-1]),
                          float(delay), int(cap)-len(requests), pending, history,
                          record.dataset, video_provider.at(now) if video_provider else None)
        action = bool(policy(context))
        if action:
            if len(requests) >= cap:
                raise ValueError('Policy requested beyond its budget')
            if queue and not multiple_pending:
                raise ValueError('Policy requested while single-request channel occupied')
            rid = len(requests)
            requests.append(dict(request_id=rid, sample_time=now,
                                 arrival_time=now+delay, cost=float(cost),
                                 delivered=False, delivered_at=None, first_usable_score_time=None))
            heapq.heappush(queue, (now+delay, i, rid))
            history = deliver(now, history)
        prediction = np.asarray(predictor(history, now), float)
        if prediction.shape != (2, 3) or not np.isfinite(prediction).all():
            raise ValueError('Predictor must return finite (2,3) positions')
        prediction = prediction.copy()
        predictions.append(prediction)
        direct = bool(abs(history.time[-1]-now) <= EPS)
        steps.append(dict(time=now, last_source_time=float(history.time[-1]),
                          age_s=max(0., now-float(history.time[-1])),
                          requested=action, requests=len(requests),
                          in_flight=len(queue), current_direct=direct))
        if observer is not None:
            # The context matches the post-action information used to predict.
            post_pending = tuple(Pending(rid, float(t[ix]), float(arrival))
                                 for arrival, ix, rid in sorted(queue))
            observer(Context(now, window.origin, window.end, float(t[window.stop-1]),
                             float(delay), int(cap)-len(requests), post_pending,
                             history, record.dataset, context.video), prediction.copy())
    # Resolve arrivals after the last sampled score but before the window ends.
    # They count as timely responses but cannot improve earlier predictions.
    history = deliver(window.end, history, at_score=False)
    for row in requests:
        row['within_window'] = bool(row['arrival_time'] <= window.end + EPS)
        row['within_scoring_period'] = bool(row['arrival_time'] <= t[window.stop-1] + EPS)
    selection = slice(window.origin_index+1, window.stop)
    predictions = np.asarray(predictions)
    reference = points[selection].mean(axis=2)
    errors = np.linalg.norm(predictions-reference, axis=-1)
    valid = ~bad[selection] & np.isfinite(reference).all(axis=-1)
    direct = np.array([s['current_direct'] for s in steps])
    unobserved = valid & ~direct[:, None]
    def mean_or_nan(values):
        return float(np.mean(values)) if len(values) else float('nan')
    metrics = dict(mean_error=mean_or_nan(errors[valid]),
                   unobserved_error=mean_or_nan(errors[unobserved]),
                   p95_error=float(np.quantile(errors[valid], .95)) if valid.any() else float('nan'),
                   max_error=float(np.max(errors[valid])) if valid.any() else float('nan'),
                   scored_hand_times=int(valid.sum()),
                   unobserved_hand_times=int(unobserved.sum()),
                   current_direct_hand_times=int((valid & direct[:, None]).sum()),
                   requests=len(requests), total_cost=float(len(requests)*cost),
                   latency_sum_s=float(len(requests)*delay),
                   waiting_union_s=waiting_union(requests, window.origin, window.end),
                   responses_within_window=sum(r['within_window'] for r in requests),
                   responses_within_scoring_period=sum(r['within_scoring_period'] for r in requests),
                   late_responses=sum(not r['within_window'] for r in requests))
    return dict(metrics=metrics, predictions=predictions, errors=errors,
                valid=valid, steps=steps, requests=requests,
                times=t[selection].copy())


class NoRequests:
    def __call__(self, context):
        return False


class Periodic:
    """Phase-controlled periodic requests on the first available raw timestamp.

    A missed nominal slot is advanced once; duplicate slots do not cause extra
    requests at a single time. Responses must arrive by the last scored sample.
    """
    def __init__(self, cap, phase=.5, single_pending=False):
        if cap < 0 or not 0 <= phase < 1:
            raise ValueError('Invalid periodic specification')
        self.cap, self.phase, self.single_pending = cap, phase, single_pending
        self.next_slot = 0

    def __call__(self, context):
        if self.cap == 0 or context.remaining <= 0:
            return False
        if self.single_pending and context.pending:
            return False
        useful = context.last_score_time-context.delay-context.origin
        if useful < 0 or context.now+context.delay > context.last_score_time+EPS:
            return False
        target = context.origin+(self.next_slot+self.phase)*useful/self.cap
        if self.next_slot < self.cap and context.now+EPS >= target:
            # Skip earlier coincident slots after taking this one sample.
            self.next_slot += 1
            while (self.next_slot < self.cap and
                   context.origin+(self.next_slot+self.phase)*useful/self.cap <= context.now+EPS):
                self.next_slot += 1
            return True
        return False
