"""Geometric and time-information invariants for the sparse expert cache."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from shuwei_c.q2_replay import History, readonly
from shuwei_c.q2_features import CausalExperts, transform_features


def history():
    t = np.array([0., .1, .3, .6, .95])
    rng = np.random.default_rng(20)
    shape = rng.normal(size=(2, 21, 3))*.01
    shape -= shape.mean(axis=1, keepdims=True)
    p = np.zeros((len(t), 2, 21, 3))
    p[:, :, :, 0] = t[:, None, None]
    p[:, 1, :, 1] = .2
    p += shape[None]
    return History(readonly(t), readonly(p), readonly(np.zeros((len(t), 2), bool)),
                   {'head': {'time': readonly(t), 'position': readonly(np.zeros((len(t), 1, 3)))}})


def test_irregular_intervals_and_zero_horizon():
    h = history(); e = CausalExperts('EgoDemo')
    pack = e.at(h, 1.1)
    np.testing.assert_allclose(pack['experts'][:, 0, 1:6], .15, atol=1e-7)
    np.testing.assert_allclose(pack['experts'][:, 1:, 1:6], 0., atol=1e-7)
    np.testing.assert_allclose(e.at(h, .95)['experts'], 0., atol=1e-7)


def test_translation_rotation_equivariance_and_invariant_features():
    h = history()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([3., -2., 1.])
    moved = History(h.time, readonly(h.points@rotation.T+translation), h.bad,
                    {'head': {'time': h.time, 'position': readonly(h.auxiliary['head']['position']@rotation.T+translation)}})
    a = CausalExperts('EgoDemo').at(h, 1.2)
    b = CausalExperts('EgoDemo').at(moved, 1.2)
    np.testing.assert_allclose(a['features'], b['features'], atol=1e-6)
    np.testing.assert_allclose(np.einsum('ij,hjk->hik', rotation, a['experts']), b['experts'], atol=1e-6)
    np.testing.assert_allclose(a['anchor']@rotation.T+translation, b['anchor'], atol=1e-9)


def test_fold_scale_only_changes_dimensional_features():
    a = CausalExperts('EgoDemo').at(history(), 1.2)
    x = transform_features(a['features'], a['feature_names'], np.ones(2))
    y = transform_features(a['features'], a['feature_names'], np.ones(2)*2)
    for j, name in enumerate(a['feature_names']):
        if name.startswith(('length_', 'speed_')):
            np.testing.assert_allclose(np.expm1(x[:, j]), 2*np.expm1(y[:, j]), rtol=1e-5, atol=1e-7)
        else:
            np.testing.assert_array_equal(x[:, j], y[:, j])
