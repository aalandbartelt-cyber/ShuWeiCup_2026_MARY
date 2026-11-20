"""Causal numerical baselines for Q2; exact source-time delayed assimilation."""
import numpy as np
from .motion import predict, update


class Kinematic:
    def __init__(self, method='constant', damping=None, lag=None):
        if method not in ('constant', 'linear', 'damped'):
            raise ValueError(method)
        self.method, self.damping, self.lag = method, damping, lag
        self._history = None

    def __call__(self, history, now):
        if history is not self._history:
            self._history = history
            c = history.centres
            self._position, self._velocity, self._times = [], [], []
            for hand in range(2):
                good = np.flatnonzero(~history.bad[:, hand] & np.isfinite(c[:, hand]).all(axis=-1))
                if not len(good):
                    good = np.flatnonzero(np.isfinite(c[:, hand]).all(axis=-1))
                if not len(good):
                    raise ValueError('No finite received hand position')
                last = good[-1]
                before = good[-2] if len(good) > 1 else last
                if self.lag is not None and len(good) > 1:
                    desired = history.time[last]-self.lag
                    k = np.searchsorted(history.time[good], desired, side='right')-1
                    before = good[max(0, min(k, len(good)-2))]
                velocity = (c[last, hand]-c[before, hand])/max(history.time[last]-history.time[before], 1e-9)
                self._position.append(c[last, hand]); self._velocity.append(velocity)
                self._times.append(history.time[last])
            self._position = np.array(self._position)
            self._velocity = np.array(self._velocity)
            self._times = np.array(self._times)
        age = np.maximum(now-self._times, 0.)
        if self.method == 'constant':
            delta = np.zeros(2)
        elif self.method == 'damped':
            if self.damping is None or self.damping <= 0:
                raise ValueError('Positive damping time required')
            delta = self.damping*(-np.expm1(-age/self.damping))
        else:
            delta = age
        return self._position+self._velocity*delta[:, None]


class Kalman:
    """All updates occur at source times; predictions do not mutate the filter."""
    def __init__(self, noise, quality=True):
        self.noise, self.quality = dict(noise), quality
        self._history = None
        self._last_time = None

    @classmethod
    def from_prefix(cls, noise, state, covariance, source_time, quality=True):
        result = cls(noise, quality)
        result._state = np.array(state, copy=True)
        result._cov = np.array(covariance, copy=True)
        result._last_time = float(source_time)
        return result

    def __call__(self, history, now):
        if history is not self._history:
            centres = history.centres
            if self._last_time is None:
                self._state = np.zeros((2, 2, 3))
                self._state[:, 0] = centres[0]
                self._cov = np.tile(np.diag([self.noise['r'], self.noise['acceleration']]), (2, 1, 1))
                self._last_time = float(history.time[0])
                start = 0
            else:
                start = np.searchsorted(history.time, self._last_time, side='right')
            for i in range(start, len(history.time)):
                dt = float(history.time[i]-self._last_time)
                self._state, self._cov = predict(self._state, self._cov, dt, self.noise)
                self._state, self._cov = update(self._state, self._cov, centres[i],
                                               history.bad[i], self.noise, self.quality)
                self._last_time = float(history.time[i])
            self._history = history
        state, _ = predict(self._state, self._cov, now-self._last_time, self.noise)
        return state[:, 0]


def make_baseline(name, noise):
    if name == 'kalman':
        return Kalman(noise)
    if name == 'kalman_no_quality':
        return Kalman(noise, quality=False)
    if name.startswith('damped_'):
        return Kinematic('damped', damping=float(name.split('_')[1]))
    return Kinematic(name)
