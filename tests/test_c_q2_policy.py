"""Check counterfactual-label boundaries and causal observation-value inputs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from test_c_q2_replay import record, window
from shuwei_c.q2_replay import replay, NoRequests, Periodic
from shuwei_c.q2_predictors import Kinematic
from shuwei_c.q2_policy import DecisionRecorder, ValueFeatures, ValuePolicy, counterfactual_pair


def test_future_mutation_changes_labels_not_current_value_features():
    a, b = record(), record()
    for side in ('left', 'right'):
        b.streams[side]['position'][12:] += .8
    recorders = []
    for rec in (a, b):
        policy = DecisionRecorder(NoRequests())
        replay(rec, window(), Kinematic(), policy, 2, .1)
        recorders.append(policy)
    ca, prefix = recorders[0].states[0]
    cb, _ = recorders[1].states[0]
    xa, names = ValueFeatures('EgoDemo', .1)(ca)
    xb, _ = ValueFeatures('EgoDemo', .1)(cb)
    np.testing.assert_array_equal(xa, xb)
    assert not any('reference' in n or 'target' in n or 'error' in n for n in names)
    la = counterfactual_pair(a, window(), ca, prefix, 2, .1, .1)
    lb = counterfactual_pair(b, window(), cb, prefix, 2, .1, .1)
    assert la['error_gain'] != lb['error_gain']


def test_value_policy_enforces_budget_and_return_deadline():
    class AlwaysPositive:
        def predict(self, x):
            return np.ones(len(x))
    r = replay(record(), window(), Kinematic(), ValuePolicy(AlwaysPositive(), 'EgoDemo', .1), 2, .3)
    assert r['metrics']['requests'] == 2
    assert all(q['within_scoring_period'] for q in r['requests'])
    assert r['steps'][1]['in_flight'] == 2


def test_counterfactuals_preserve_issued_pending_prefix():
    recorder = DecisionRecorder(Periodic(4, phase=0.))
    replay(record(), window(), Kinematic(), recorder, 4, .3)
    candidates = [(ctx, prefix) for ctx, prefix in recorder.states if prefix and ctx.pending]
    assert candidates
    ctx, prefix = candidates[0]
    label = counterfactual_pair(record(), window(), ctx, prefix, 4, .3, .1)
    assert label['wait_requests'] <= 4 and label['request_requests'] <= 4
    assert np.isfinite(label['error_gain'])
