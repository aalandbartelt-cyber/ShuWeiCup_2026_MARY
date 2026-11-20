"""Meaningful event/queue boundary tests for visible threshold policies."""
from dataclasses import replace
import numpy as np
from shuwei_c.q2_replay import Context, History, Pending
from shuwei_c.q2_threshold import VisibleThreshold


def context(now=.2):
    p = np.zeros((2, 2, 21, 3)); p[1, :, :, 0] = .1
    h = History(np.array([-.1, 0.]), p, np.zeros((2, 2), bool))
    return Context(now, 0., 1., 1., .1, 2, (), h, 'synthetic')


def test_age_queue_and_repeated_tick():
    policy = VisibleThreshold('age', 1., 2, .1)
    c = context(.3)
    assert policy(c)
    assert not policy(c)
    pending = (Pending(0, .3, .4),)
    assert not policy(replace(c, now=.39, remaining=1, pending=pending))


def test_motion_scale_and_minimum_gap():
    policy = VisibleThreshold('motion', .5, 2, .1, 2.)
    assert policy(context(.1))
    assert not policy(context(.25))
    assert policy(context(.3))
    slow = VisibleThreshold('motion', .5, 2, 1.)
    assert not slow(context(.1))


def test_budget_deadline_and_qualified_history():
    c = context(.2)
    policy = VisibleThreshold('motion', .5, 2, .1)
    assert not policy(replace(c, remaining=0))
    assert not policy(replace(c, now=.91))
    p = c.history.points.copy(); p[1] = 1000.
    bad = c.history.bad.copy(); bad[1] = True
    # The huge flagged disclosure must not act as a speed observation.
    h = History(c.history.time, p, bad)
    assert not policy(replace(c, history=h))
