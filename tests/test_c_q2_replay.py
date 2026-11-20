"""Synthetic invariants for delayed Q2 information, clocks and independent metrics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from shuwei_c.data import Recording
from shuwei_c.q2_replay import (Window, replay, select_windows, Periodic,
                               NoRequests, waiting_union)
from shuwei_c.q2_predictors import Kinematic, Kalman


def record():
    t = np.arange(31)/10
    points = np.zeros((len(t), 2, 21, 3))
    points[:, :, :, 0] = t[:, None, None]
    streams = {
        'left': dict(time=t.copy(), position=points[:, 0].copy()),
        'right': dict(time=t.copy(), position=points[:, 1].copy()),
        'quality': dict(time=t.copy(), bad=np.zeros((len(t), 2), bool)),
        'head': dict(time=t.copy(), position=np.zeros((len(t), 1, 3)))
    }
    return Recording('private', 'EgoDemo', 'private_environment', 'private_operator',
                     'train', Path('private_path'), streams)


def window(end=2.):
    return Window('key', 'private', 'EgoDemo', 'private_environment', 'train',
                  10, round(end*10)+1, 1., end-1.)


class Schedule:
    def __init__(self, times):
        self.times = times

    def __call__(self, context):
        return any(abs(context.now-t) < 1e-8 for t in self.times)


def test_linear_motion_and_error_definition():
    r = replay(record(), window(), Kinematic('linear'), NoRequests(), 0, .1)
    np.testing.assert_allclose(r['predictions'][:, :, 0], r['times'][:, None]*np.ones((1, 2)), atol=1e-12)
    assert r['metrics']['mean_error'] < 1e-12
    r = replay(record(), window(), Kinematic(), NoRequests(), 0, .1)
    assert r['metrics']['mean_error'] == pytest.approx(.55)
    assert r['metrics']['scored_hand_times'] == 20


def test_future_positions_flags_head_unavailable_until_actual_arrival():
    a, b = record(), record()
    for side in ('left', 'right'):
        b.streams[side]['position'][11:] += 100
    b.streams['quality']['bad'][11:] = True
    b.streams['head']['position'][11:] = 888
    seen = []
    def observe(ctx, prediction):
        seen.append(ctx)
        assert not hasattr(ctx, 'record') and not hasattr(ctx, 'reference')
        assert ctx.history.time[-1] <= ctx.now+1e-9
        assert not ctx.history.points.flags.writeable
    ra = replay(a, window(), Kinematic(), Schedule([1.1]), 1, .3)
    rb = replay(b, window(), Kinematic(), Schedule([1.1]), 1, .3, observer=observe)
    before = ra['times'] < 1.4-1e-9
    np.testing.assert_array_equal(ra['predictions'][before], rb['predictions'][before])
    for ctx in seen:
        if ctx.now < 1.4-1e-9:
            assert ctx.history.auxiliary['head']['time'][-1] == 1.
            assert not ctx.history.bad.any()
        else:
            assert ctx.history.time[-1] == pytest.approx(1.1)
            assert ctx.history.auxiliary['head']['time'][-1] == pytest.approx(1.1)
            assert ctx.history.bad[-1].all()


def test_request_returns_source_sample_not_response_sample():
    r = replay(record(), window(), Kinematic(), Schedule([1.1]), 1, .3)
    i = np.flatnonzero(np.isclose(r['times'], 1.4))[0]
    np.testing.assert_allclose(r['predictions'][i, :, 0], 1.1)
    assert r['steps'][i]['age_s'] == pytest.approx(.3)


def test_zero_delay_used_now_and_unobserved_denominator():
    r = replay(record(), window(), Kinematic(), Schedule([1.1]), 1, 0.)
    assert r['metrics']['current_direct_hand_times'] == 2
    assert r['metrics']['unobserved_hand_times'] == 18
    assert r['metrics']['mean_error'] == pytest.approx(.45)
    assert r['metrics']['unobserved_error'] == pytest.approx(.5)
    assert r['requests'][0]['first_usable_score_time'] == pytest.approx(1.1)


def test_inflight_union_does_not_double_count_and_cost_is_immediate():
    r = replay(record(), window(), Kinematic(), Schedule([1.1, 1.2]), 2, .3)
    assert r['metrics']['waiting_union_s'] == pytest.approx(.4)
    assert r['metrics']['latency_sum_s'] == pytest.approx(.6)
    assert r['metrics']['total_cost'] == 2
    assert r['steps'][1]['in_flight'] == 2
    assert r['steps'][1]['requests'] == 2


def test_late_return_charged_and_censored():
    r = replay(record(), window(), Kinematic(), Schedule([1.9]), 1, .3)
    assert r['metrics']['late_responses'] == 1
    assert r['metrics']['waiting_union_s'] == pytest.approx(.1)
    assert r['metrics']['total_cost'] == 1
    np.testing.assert_allclose(r['predictions'][:, :, 0], 1.)


def test_boundary_arrival_usable_and_single_channel_enforced():
    r = replay(record(), window(), Kinematic(), Schedule([1.7]), 1, .3)
    np.testing.assert_allclose(r['predictions'][-1, :, 0], 1.7)
    assert r['metrics']['responses_within_window'] == 1
    with pytest.raises(ValueError, match='single-request'):
        replay(record(), window(), Kinematic(), Schedule([1.1, 1.2]), 2, .3,
               multiple_pending=False)


def test_budget_and_prediction_validation():
    with pytest.raises(ValueError, match='budget'):
        replay(record(), window(), Kinematic(), Schedule([1.1]), 0, .1)
    with pytest.raises(ValueError, match='finite'):
        replay(record(), window(), lambda h, t: np.full((2, 3), np.nan), NoRequests(), 0, .1)


def test_periodic_does_not_issue_useless_or_duplicate_requests():
    for cap in (0, 1, 2, 4):
        for delay in (0., .1, .3, 2.):
            r = replay(record(), window(), Kinematic(), Periodic(cap), cap, delay)
            times = [q['sample_time'] for q in r['requests']]
            assert len(times) == len(set(times)) <= cap
            assert all(q['within_scoring_period'] for q in r['requests'])


def test_non_grid_arrival_uses_first_following_score_time():
    r = replay(record(), window(), Kinematic(), Schedule([1.1]), 1, .15)
    assert r['requests'][0]['arrival_time'] == pytest.approx(1.25)
    assert r['requests'][0]['first_usable_score_time'] == pytest.approx(1.3)
    assert r['metrics']['waiting_union_s'] == pytest.approx(.15)


def test_arrival_after_last_score_but_before_end_is_delivered_and_serializable():
    import json
    w = Window('key', 'private', 'EgoDemo', 'private_environment', 'train',
               10, 21, 1., 1.05)
    result = replay(record(), w, Kinematic(), Schedule([1.8]), 1, .23)
    request = result['requests'][0]
    assert request['within_window'] and request['delivered']
    assert not request['within_scoring_period']
    assert request['delivered_at'] == pytest.approx(2.03)
    assert request['first_usable_score_time'] is None
    np.testing.assert_allclose(result['predictions'][:, :, 0], 1.)
    assert json.loads(json.dumps(result['requests'])) == result['requests']


def test_kalman_delayed_source_assimilation_matches_visible_history_filter():
    noise = dict(r=.01, acceleration=.5, hand_scale=.1)
    traces = []
    result = replay(record(), window(), Kalman(noise), Schedule([1.2, 1.4]), 2, .2,
                    observer=lambda ctx, pred: traces.append((ctx, pred)))
    for ctx, pred in traces:
        reference = Kalman(noise)(ctx.history, ctx.now)
        np.testing.assert_allclose(pred, reference, atol=1e-12)
    assert result['metrics']['responses_within_window'] == 2


def test_quality_selection_is_private_and_reference_not_interpolated():
    config = dict(main_windows_s=[.2, .5], stress_windows_s=[2.], warmup_s=1., window_stride_s=1.)
    a = record()
    windows, audit = select_windows(a, config)
    assert len(windows) == 5
    assert all(w.stop > w.origin_index+1 for w in windows)
    a.streams['quality']['bad'][11, 0] = True
    windows2, audit2 = select_windows(a, config)
    assert len(windows2) == 2
    assert sum(r['reason'] == 'flagged_reference' for r in audit2) == 3
    assert sum(r['reason'] == 'record_end' for r in audit) == 4


def test_union_matches_independent_dense_numerical_integral():
    requests = [dict(sample_time=1.05, arrival_time=1.35),
                dict(sample_time=1.2, arrival_time=1.5),
                dict(sample_time=1.8, arrival_time=2.2)]
    grid = np.arange(1.00005, 2., .0001)
    active = np.array([any(r['sample_time'] <= t < r['arrival_time'] for r in requests) for t in grid])
    assert waiting_union(requests, 1., 2.) == pytest.approx(active.mean(), abs=1e-8)


def test_video_is_only_supplied_by_prefix_provider():
    class Provider:
        def at(self, now):
            return dict(timestamp=now-.01, value=3.)
    contexts = []
    replay(record(), window(), Kinematic(), NoRequests(), 0, .1,
           video_provider=Provider(), observer=lambda ctx, pred: contexts.append(ctx))
    assert all(ctx.video['timestamp'] < ctx.now for ctx in contexts)
