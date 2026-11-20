"""Causal age and visible-motion threshold comparators for Q2.

AI-assisted: OpenAI Codex, 2026-11-20; version release date unavailable.
No hidden references, learned radius, or recording handles enter these policies.
"""
import math
import numpy as np
from .q2_replay import EPS


class VisibleThreshold:
    """Threshold source age or visible speed times source age.

    Pending source timestamps reset the expected source age without revealing
    packet contents. Speed uses the last two qualified disclosed observations.
    Age thresholds are multiples of (last score-origin-delay)/(budget+1).
    Motion thresholds are in units of the fitting environments' hand scale.
    """
    def __init__(self, kind, threshold, cap, scale, minimum_gap_factor=1.):
        if kind not in ('age', 'motion'):
            raise ValueError('Unknown threshold kind')
        if threshold <= 0 or scale <= 0 or minimum_gap_factor < 0:
            raise ValueError('Invalid threshold, scale or spacing')
        if int(cap) != cap or cap < 0:
            raise ValueError('Nonnegative integer budget required')
        self.kind, self.threshold, self.cap = kind, threshold, int(cap)
        self.scale, self.minimum_gap_factor = scale, minimum_gap_factor
        self._history = None
        self.last_request = -math.inf

    def __call__(self, context):
        if context.remaining <= 0 or context.now + context.delay > context.last_score_time + EPS:
            return False
        if context.now <= self.last_request + EPS:
            return False
        if context.now - self.last_request + EPS < self.minimum_gap_factor * context.delay:
            return False
        if context.history is not self._history:
            self._history = context.history
            h, centres = context.history, context.history.centres
            source, speed = [], []
            for hand in range(2):
                good = np.flatnonzero(~h.bad[:, hand] & np.isfinite(centres[:, hand]).all(axis=-1))
                if not len(good):
                    good = np.flatnonzero(np.isfinite(centres[:, hand]).all(axis=-1))
                if not len(good):
                    raise ValueError('No finite disclosed position')
                last = good[-1]
                before = good[-2] if len(good) > 1 else last
                source.append(h.time[last])
                speed.append(np.linalg.norm(centres[last, hand] - centres[before, hand]) /
                             max(h.time[last] - h.time[before], EPS))
            self.source, self.speed = np.asarray(source), np.asarray(speed)
        source = self.source
        if context.pending:
            source = np.maximum(source, max(p.sample_time for p in context.pending))
        age = np.maximum(context.now - source, 0.)
        if self.kind == 'age':
            useful = max(context.last_score_time - context.origin - context.delay, 0.)
            trigger = age.mean() + EPS >= self.threshold * useful / (self.cap + 1)
        else:
            trigger = np.mean(age * self.speed) / self.scale + EPS >= self.threshold
        if trigger:
            self.last_request = context.now
        return bool(trigger)
