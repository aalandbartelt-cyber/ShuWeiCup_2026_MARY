"""Meaningful calibration invariants, grouping, held-out labels and path coverage."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import pandas as pd
import pytest

from shuwei_c.q2_calibration import (RadiusCalibrator, AgeBinnedRadius, hierarchical_weights,
                                    weighted_quantile, oof_predictions,
                                    evaluate_radii, suggest_horizon,
                                    evaluate_paths, evaluate_paths_long, evaluate_suggestions)


def metadata(n, environments=('a', 'b', 'c', 'd')):
    return pd.DataFrame(dict(environment=[environments[i % len(environments)] for i in range(n)],
                             episode=[f'r{i % len(environments)}' for i in range(n)],
                             window_id=[f'w{i // (len(environments)*2)}' for i in range(n)],
                             hand=[(i//len(environments)) % 2 for i in range(n)],
                             scenario=['b0_d0.1']*n))


def test_hierarchical_weights_ignore_record_lengths_and_keep_scenarios_distinct():
    rows = []
    for env, episode, window, scenario, count in [
        ('a', 'r1', 'w1', 'A', 100), ('a', 'r1', 'w1', 'B', 1),
        ('a', 'r1', 'w2', 'A', 1), ('a', 'r1', 'w2', 'B', 1),
        ('a', 'r2', 'w3', 'A', 1), ('b', 'r3', 'w4', 'A', 1)]:
        for hand in (0, 1):
            rows.extend(dict(environment=env, episode=episode, window_id=window,
                             scenario=scenario, hand=hand) for _ in range(count))
    meta = pd.DataFrame(rows)
    meta['weight'] = hierarchical_weights(meta)
    assert meta.groupby('environment').weight.sum().to_dict() == pytest.approx({'a': .5, 'b': .5})
    assert meta[meta.episode == 'r1'].groupby(['window_id', 'scenario']).weight.sum().to_numpy() == pytest.approx([.0625]*4)
    assert meta[meta.episode == 'r2'].weight.sum() == pytest.approx(.25)
    assert meta[meta.episode == 'r3'].weight.sum() == pytest.approx(.5)


def test_weighted_quantile_is_not_frame_frequency_quantile():
    meta = metadata(102, environments=('a',))
    meta['episode'] = ['long']*100+['short']*2
    meta['window_id'] = 'w'
    errors = np.r_[np.zeros(100), np.ones(2)*10]
    weights = hierarchical_weights(meta)
    assert weighted_quantile(errors, .75, weights) == 10
    assert np.quantile(errors, .75) == 0


def test_static_prediction_and_environment_disjoint_correction():
    model = RadiusCalibrator('static', levels=(.5, .9)).fit(np.ones((8, 1)), np.arange(8), metadata(8))
    assert np.diff(model.predict(np.zeros((3, 1))), axis=1).min() >= 0
    with pytest.raises(ValueError, match='overlap'):
        model.calibrate(np.ones((8, 1)), np.arange(8), metadata(8))
    cal = metadata(4, environments=('independent',))
    model.calibrate(np.ones((4, 1)), np.ones(4)*12, cal)
    np.testing.assert_allclose(model.predict(np.ones((2, 1))), 12.)


def test_oof_predictions_do_not_depend_on_their_held_labels():
    meta = metadata(80)
    features = np.zeros((80, 2))
    errors = np.arange(80, dtype=float)/10
    environments = sorted(meta.environment.unique())
    folds = [dict(fold=i, training_environments=[e for e in environments if e != held],
                  held_environments=[held]) for i, held in enumerate(environments)]
    before = oof_predictions(features, errors, meta, folds, method='static')
    changed = errors.copy(); changed[meta.environment == 'a'] = 1e6
    after = oof_predictions(features, changed, meta, folds, method='static')
    np.testing.assert_array_equal(before['radii'][meta.environment == 'a'], after['radii'][meta.environment == 'a'])
    assert all(not (set(f['fitting_environments']) & set(f['held_environments'])) for f in before['folds'])
    with pytest.raises(ValueError, match='cover all'):
        oof_predictions(features, errors, meta, folds[:-1], method='static')
    with pytest.raises(ValueError, match='more than once'):
        oof_predictions(features, errors, meta, folds+[folds[0]], method='static')


def test_oof_inner_calibration_is_removed_from_fitting():
    meta = metadata(24, environments=('a', 'b', 'c'))
    folds = []
    for i, held in enumerate(['a', 'b', 'c']):
        training = [e for e in ['a', 'b', 'c'] if e != held]
        folds.append(dict(training_environments=training, held_environments=[held],
                          calibration_environments=training[:1]))
    result = oof_predictions(np.ones((24, 1)), np.arange(24)/24, meta, folds, method='static')
    assert np.isfinite(result['radii']).all()
    assert all(len(f['fitting_environments']) == len(f['calibration_environments']) == 1 for f in result['folds'])


def test_conditional_model_learns_heteroscedastic_radii_and_serializes(tmp_path):
    rng = np.random.default_rng(123)
    x = rng.integers(0, 2, (1600, 1)).astype(float)
    errors = (.1+.9*x[:, 0])*rng.uniform(.6, 1.4, len(x))
    before = np.random.get_state()
    model = RadiusCalibrator(parameters=dict(num_leaves=7, min_child_samples=20,
                                             learning_rate=.08, n_estimators=80))
    model.fit(x, errors, metadata(len(x)))
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    test_x = np.tile([[0.], [1.]], (200, 1))
    test_errors = (.1+.9*test_x[:, 0])*rng.uniform(.6, 1.4, len(test_x))
    test_meta = metadata(len(test_x), environments=('held',))
    conditional = model.predict(test_x)
    static = RadiusCalibrator('static').fit(x, errors, metadata(len(x))).predict(test_x)
    c_score = evaluate_radii(test_meta, test_errors, conditional).pinball.mean()
    s_score = evaluate_radii(test_meta, test_errors, static).pinball.mean()
    assert c_score < .5*s_score
    assert conditional[0, 2] < .2 and conditional[1, 2] > 1.
    assert np.all(np.diff(conditional, axis=1) >= 0)
    model.save(tmp_path)
    np.testing.assert_array_equal(RadiusCalibrator.load(tmp_path).predict(test_x), conditional)


def test_feature_order_and_invalid_errors_are_rejected():
    x = pd.DataFrame({'age': [0., 1.], 'speed': [1., 2.]})
    model = RadiusCalibrator('static').fit(x, [0., 1.], metadata(2))
    with pytest.raises(ValueError, match='names/order'):
        model.predict(x[['speed', 'age']])
    with pytest.raises(ValueError, match='nonnegative'):
        RadiusCalibrator('static').fit(x, [0., -1.], metadata(2))


def test_pointwise_evaluation_separates_conditions_and_direct_returns():
    meta = metadata(8, environments=('a',))
    meta['cap'] = [0]*4+[1]*4
    meta['current_direct'] = [False]*4+[True]*4
    errors = np.r_[np.ones(4), np.zeros(4)]
    result = evaluate_radii(meta, errors, np.full((8, 1), .5), levels=(.9,), group_by=('cap',))
    assert result.query("cap == 0 and subset == 'all'").coverage.iloc[0] == 0
    assert result.query("cap == 1 and subset == 'all'").coverage.iloc[0] == 1
    empty = result.query("cap == 1 and subset == 'not_current_direct'").iloc[0]
    assert empty.rows == 0 and np.isnan(empty.coverage)


def test_horizon_uses_monotone_prefix_not_a_later_radius_dip():
    horizons = [.1, .2, .5]
    radii = [[.2, 1.2, .3], [.6, .8, 1.1], [.1, np.inf, .2]]
    suggested = suggest_horizon(horizons, radii, (.5, 1., 2.))
    np.testing.assert_array_equal(suggested, [[.1, .1, .5], [0., .2, .5], [.1, .1, .1]])
    with pytest.raises(ValueError, match='horizon'):
        suggest_horizon([.2, .1], [[1., 2.]])
    with pytest.raises(ValueError, match='radii'):
        suggest_horizon([.1, .2], [[np.nan, 2.]])


def test_ninety_percent_point_coverage_can_have_zero_path_coverage():
    meta = metadata(10, environments=('a',))
    meta['window_id'] = [f'w{i}' for i in range(10)]
    errors = np.zeros((10, 10)); errors[np.arange(10), np.arange(10)] = 2
    result = evaluate_paths(meta, errors, np.ones((10, 10)))
    assert result['summary']['pointwise_coverage'] == pytest.approx(.9)
    assert result['summary']['entire_path_coverage'] == 0.


def test_path_coverage_is_joint_over_hands_and_requires_complete_references():
    meta = metadata(4, environments=('a',))
    meta['window_id'] = ['w0', 'w0', 'w1', 'w1']
    errors = np.array([[0., 0.], [0., 2.], [np.nan, 0.], [0., 0.]])
    valid = np.isfinite(errors)
    result = evaluate_paths(meta, errors, np.ones((4, 2)), valid=valid)
    assert result['summary']['windows'] == 2
    assert result['summary']['complete_reference_windows'] == 1
    assert result['summary']['entire_path_coverage'] == 0.
    assert result['per_window'].loc[1, 'complete_reference'] == False


def test_zero_duration_is_abstention_not_automatic_success():
    meta = metadata(2, environments=('a',))
    meta['window_id'] = ['w0', 'w1']
    result = evaluate_suggestions(meta, [.1, .2], [[.1, 2.], [.1, .2]],
                                  [[0.], [.2]], tolerances=(.5,)).iloc[0]
    assert result.suggestion_rate == pytest.approx(.5)
    assert result.assessable_rate == pytest.approx(.5)
    assert result.selected_interval_within_tolerance == 1.
    assert result.mean_suggested_horizon == pytest.approx(.1)


def test_missing_reference_does_not_certify_selected_interval():
    meta = metadata(1, environments=('a',))
    result = evaluate_suggestions(meta, [.1, .2], [[.1, np.nan]], [[.2]],
                                  tolerances=(.5,), valid=[[True, False]]).iloc[0]
    assert result.suggestion_rate == 1.
    assert result.assessable_rate == 0.
    assert np.isnan(result.selected_interval_within_tolerance)
    with pytest.raises(ValueError, match='grid horizon'):
        evaluate_suggestions(meta, [.1, .2], [[.1, .2]], [[.15]], tolerances=(.5,))


def test_age_bins_use_only_fit_data_and_sparse_bin_falls_back():
    meta = metadata(12, environments=('a', 'b', 'c'))
    ages = np.r_[np.ones(6)*.02, np.ones(6)*.8]
    errors = np.r_[np.ones(6)*.1, np.ones(6)]
    model = AgeBinnedRadius().fit(ages, errors, meta)
    predicted = model.predict([.02, .8, 3.])
    np.testing.assert_allclose(predicted[0], .1)
    np.testing.assert_allclose(predicted[1], 1.)
    np.testing.assert_allclose(predicted[2], model.global_radius)


def test_long_path_counts_detect_missing_and_duplicate_rows():
    meta = metadata(8, environments=('a',))
    meta['window_id'] = ['w0']*4+['w1']*4
    meta['time_s'] = [1., 1., 2., 2.]*2
    meta['expected_hand_times'] = [4]*8
    errors = np.zeros(8); errors[3] = 2
    paths = evaluate_paths_long(meta, errors, np.ones((8, 1)), levels=(.9,))
    assert paths.entire_path_covered.tolist() == [0., 1.]
    incomplete = evaluate_paths_long(meta.iloc[:-1], errors[:-1], np.ones((7, 1)), levels=(.9,))
    assert incomplete.complete_reference.tolist() == [True, False]
    assert np.isnan(incomplete.entire_path_covered.iloc[1])
    with pytest.raises(ValueError, match='Repeated'):
        evaluate_paths_long(pd.concat([meta, meta.iloc[:1]], ignore_index=True),
                            np.r_[errors, 0.], np.ones((9, 1)), levels=(.9,))
