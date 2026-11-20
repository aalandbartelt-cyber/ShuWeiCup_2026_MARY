"""Sparse-history motion experts and visible features; no targets or record lookup.

Raw-coordinate features are stored before fold-specific scaling. This permits
reusing the same causal feature cache without fitting a scale on held-out data.
"""
import numpy as np

DATASETS = ('EgoDemo', 'EgoPro', 'EgoStandard')
VELOCITY_LAGS = (1/30, .1, .2, .4, .8)
DAMPING = (.05, .1, .2, .5, 1.)
EXPERT_NAMES = (['hold']+[f'velocity_{x:g}' for x in VELOCITY_LAGS]
                +[f'damped_fast_{x:g}' for x in DAMPING]
                +[f'damped_slow_{x:g}' for x in (.1, .3, .8)]
                +['quadratic_5', 'quadratic_9'])


class CausalExperts:
    def __init__(self, dataset):
        if dataset not in DATASETS:
            raise ValueError('Unknown declared dataset')
        self.dataset = dataset
        self._history = None

    def prepare(self, history):
        if history is self._history:
            return
        self._history = history
        self.t = history.time
        self.centres = history.centres
        self.anchor = np.zeros((2, 3)); self.anchor_time = np.zeros(2)
        self.velocity = np.zeros((2, 3, len(VELOCITY_LAGS)))
        self.polynomial = np.zeros((2, 2, 2, 3))
        self.shape = np.zeros((2, 3)); self.counts = []
        self.gaps = np.zeros((2, 3)); self.fallback = np.zeros(2)
        for hand in range(2):
            indices = np.flatnonzero(~history.bad[:, hand] & np.isfinite(self.centres[:, hand]).all(axis=-1))
            if not len(indices):
                indices = np.flatnonzero(np.isfinite(self.centres[:, hand]).all(axis=-1))
                self.fallback[hand] = 1.
            if not len(indices):
                raise ValueError('No finite hand history')
            indices = indices[-64:]
            t = self.t[indices]; c = self.centres[indices, hand]
            self.counts.append(t)
            self.anchor[hand] = c[-1]; self.anchor_time[hand] = t[-1]
            if len(t) > 1:
                diff = np.diff(t)
                self.gaps[hand] = [diff[-1], np.median(diff[-8:]), np.max(diff[-8:])]
                for k, lag in enumerate(VELOCITY_LAGS):
                    before = max(0, min(len(t)-2, np.searchsorted(t, t[-1]-lag, side='right')-1))
                    self.velocity[hand, :, k] = (c[-1]-c[before])/(t[-1]-t[before])
                for k, size in enumerate((5, 9)):
                    tt = t[-size:]-t[-1]
                    span = max(abs(tt[0]), 1e-6)
                    u = tt/span
                    if len(tt) >= 3:
                        design = np.stack([u, u*u], axis=1)
                        beta = np.linalg.lstsq(design, c[-size:]-c[-1], rcond=1e-6)[0]
                        self.polynomial[hand, k, 0] = beta[0]/span
                        self.polynomial[hand, k, 1] = beta[1]/span**2
            shape = history.points[indices[-1], hand]-c[-1]
            self.shape[hand] = np.linalg.svd(shape, compute_uv=False)
        head = history.auxiliary.get('head')
        self.head_time = float(head['time'][-1]) if head else float(self.t[-1])
        self.head_known = float(head is not None)
        self.head_position = head['position'][-1].reshape(3) if head else np.zeros(3)

    def at(self, history, now):
        self.prepare(history)
        age = np.maximum(now-self.anchor_time, 0.)
        experts = [np.zeros((2, 3))]
        experts.extend(self.velocity[:, :, k]*age[:, None] for k in range(len(VELOCITY_LAGS)))
        experts.extend(self.velocity[:, :, 0]*(tau*(-np.expm1(-age/tau)))[:, None] for tau in DAMPING)
        experts.extend(self.velocity[:, :, 2]*(tau*(-np.expm1(-age/tau)))[:, None] for tau in (.1, .3, .8))
        for k in range(2):
            experts.append(self.polynomial[:, k, 0]*age[:, None]+self.polynomial[:, k, 1]*(age*age)[:, None])
        e = np.stack(experts, axis=-1)
        x = {}
        x['age_s'] = age
        x['packet_age_s'] = np.full(2, max(now-self.t[-1], 0.))
        x['received_bad'] = history.bad[-1].astype(float)
        x['no_qualified_history'] = self.fallback
        for k, name in enumerate(('latest_gap_s', 'median_gap_s', 'maximum_gap_s')):
            x[name] = self.gaps[:, k]
        x['received_in_last_second'] = np.array([np.sum(t >= now-1.) for t in self.counts], float)
        x['qualified_history_count'] = np.array([len(t) for t in self.counts], float)
        x['past_bad_fraction'] = history.bad[-30:].mean(axis=0)
        speed = np.linalg.norm(self.velocity, axis=1)
        for k in range(len(VELOCITY_LAGS)):
            x[f'speed_{k}'] = speed[:, k]
        for k in range(1, len(VELOCITY_LAGS)):
            x[f'cos_fast_{k}'] = np.sum(self.velocity[:, :, 0]*self.velocity[:, :, k], axis=1)/(speed[:, 0]*speed[:, k]+1e-12)
        for k in range(e.shape[-1]):
            x[f'length_expert_{k}'] = np.linalg.norm(e[:, :, k], axis=1)
        for k in range(3):
            x[f'length_shape_{k}'] = self.shape[:, k]
        x['length_interhand'] = np.full(2, np.linalg.norm(self.anchor[1]-self.anchor[0]))
        x['length_head_distance'] = np.linalg.norm(self.anchor-self.head_position, axis=1)*self.head_known
        x['head_known'] = np.full(2, self.head_known)
        x['head_age_s'] = np.full(2, max(now-self.head_time, 0.))
        x['side'] = np.arange(2, dtype=float)
        for d in DATASETS:
            x['dataset_'+d] = np.full(2, float(d == self.dataset))
        return dict(features=np.stack(list(x.values()), axis=1).astype(np.float32),
                    feature_names=list(x), experts=e.astype(np.float32),
                    anchor=self.anchor.copy(), age=age.copy())


def transform_features(features, names, scale):
    result = np.asarray(features, float).copy()
    scale = np.asarray(scale).reshape(-1)
    for j, name in enumerate(names):
        if name.startswith(('length_', 'speed_')):
            result[:, j] = np.log1p(np.maximum(result[:, j]/scale, 0.))
        elif name.endswith('_s') or name in ('qualified_history_count', 'received_in_last_second'):
            result[:, j] = np.log1p(np.maximum(result[:, j], 0.))
    return result.astype(np.float32)
