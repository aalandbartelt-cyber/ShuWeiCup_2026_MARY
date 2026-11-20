"""Hand-checkable sampling bounds, edge censoring and recovery risk sets."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from shuwei_c.q2_natural import flag_runs, record_statistics, product_limit, survival_at


def test_single_frame_has_zero_lower_bound_not_zero_observed_latency():
    events = flag_runs([0., .1, .2], [False, True, False])
    assert len(events) == 1
    event = events[0]
    assert event['flagged_samples'] == 1
    assert event['full_duration_lower_s'] == 0
    assert event['full_duration_upper_s'] == pytest.approx(.2)
    assert event['detection_to_recovery_s'] == pytest.approx(.1)
    assert not event['left_censored'] and not event['right_censored']


def test_both_record_boundaries_and_unequal_sampling_intervals():
    events = flag_runs([0., .1, .3, .5, .8], [True, True, False, False, True])
    assert len(events) == 2
    left, right = events
    assert left['left_censored'] and not left['right_censored']
    assert left['full_duration_lower_s'] == pytest.approx(.1)
    assert np.isnan(left['full_duration_upper_s'])
    assert left['detection_to_recovery_s'] == pytest.approx(.3)
    assert right['right_censored'] and not right['left_censored']
    assert right['detection_followup_s'] == 0
    assert np.isnan(right['detection_to_recovery_s'])


def test_all_bad_is_double_censored_with_finite_observed_span():
    events = flag_runs([0., .1, .4], [True, True, True])
    e = events[0]
    assert e['left_censored'] and e['right_censored']
    assert e['full_duration_lower_s'] == pytest.approx(.4)
    assert np.isnan(e['full_duration_upper_s'])
    assert not e['recovery_observed']


def test_recurrence_does_not_treat_short_followup_as_stable_recovery():
    e = flag_runs([0., .1, .2, .3], [False, True, False, False])[0]
    assert np.isnan(e['recurrence_by_0.2s'])
    events = flag_runs(np.arange(8)/10, [False, True, False, True, False, False, False, False])
    assert events[0]['recurrence_by_0.2s'] == 1
    assert events[1]['recurrence_by_0.2s'] == 0
    assert np.isnan(events[1]['recurrence_by_0.5s'])


def test_accounting_exposure_uses_timestamps_not_number_of_flags():
    t, bad = np.array([0., .1, .4, 1.]), np.array([False, True, False, True])
    s = record_statistics(t, bad, flag_runs(t, bad))
    assert s['sample_flag_fraction'] == .5
    assert s['flagged_left_hold_s'] == pytest.approx(.3)
    assert s['good_left_hold_s'] == pytest.approx(.7)
    assert s['record_span_s'] == 1
    assert s['observed_onsets'] == 2


def test_product_limit_recovery_censor_ties_and_unobserved_tail():
    curve = product_limit([1., 2., 2., 3.], [True, True, False, False])
    assert curve[0]['at_risk'] == 4
    assert curve[0]['probability_not_yet_recovered'] == .75
    assert curve[1]['at_risk'] == 3
    assert curve[1]['recoveries'] == 1 and curve[1]['censored'] == 1
    assert curve[1]['probability_not_yet_recovered'] == pytest.approx(.5)
    assert survival_at(curve, .5) == 1
    assert survival_at(curve, 2.5) == pytest.approx(.5)
    assert np.isnan(survival_at(curve, 4.))


def test_product_limit_matches_empirical_survival_without_censoring():
    d = np.array([.1, .1, .2, .4, .7])
    c = product_limit(d, np.ones(len(d), bool))
    for horizon in (0, .1, .15, .4, 2):
        assert survival_at(c, horizon) == pytest.approx(float(np.mean(d > horizon)))


def test_no_event_and_invalid_time_series():
    assert flag_runs([0, 1, 2], [False]*3) == []
    assert product_limit([], []) == []
    assert np.isnan(survival_at([], 1))
    with pytest.raises(ValueError):
        flag_runs([0, 0, 1], [False]*3)
    with pytest.raises(ValueError):
        product_limit([-.1], [False])
