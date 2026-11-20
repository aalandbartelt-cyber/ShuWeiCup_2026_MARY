"""Causal sequence construction and genuine masked-time encoder checks."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
import torch
from shuwei_c.q2_replay import History, readonly
from shuwei_c.q2_sequence import history_sequence, DecayGRU, AuthorCRU, SequenceExpertGate, SequencePredictor


def history(shift=0.):
    time = np.array([0., .1, .2, .5])
    points = np.zeros((4, 2, 21, 3))
    points[:, :, :, 0] = time[:, None, None]
    points[:, 1, :, 1] = 1
    points += shift
    bad = np.zeros((4, 2), bool)
    bad[2, 0] = True
    head = dict(time=readonly([0., .2]), position=readonly(np.array([[[0, 0, 2]], [[.2, 0, 2]]])+shift))
    return History(readonly(time), readonly(points), readonly(bad), {'head': head})


def test_sequence_uses_received_good_samples_and_records_age():
    h = history()
    anchor = h.centres[-1]
    pack = history_sequence(h, anchor, steps=6, step_s=.1)
    # Left quality at .2 is bad, so its source event keeps .1.
    assert pack['values'][0, 4, 0] == pytest.approx(-.4)
    assert pack['elapsed'][0, 4, 0] == pytest.approx(.1)
    assert not pack['mask'][0, 4, 0]
    assert pack['mask'][0, 5, 0]
    assert pack['last_source_time'] == .5
    assert pack['elapsed'][0, 5, 6] == pytest.approx(.3)
    np.testing.assert_allclose(pack['grid_time'], [-.2, -.1, 0., .1, .2, .5])
    assert pack['step_dt'][0, -1] == pytest.approx(.3)


def test_sequence_is_translation_invariant_and_has_no_future_head():
    a, b = history(), history(123.)
    pa = history_sequence(a, a.centres[-1], steps=6, step_s=.1)
    pb = history_sequence(b, b.centres[-1], steps=6, step_s=.1)
    np.testing.assert_allclose(pa['values'], pb['values'], atol=1e-6)
    changed = history()
    head = dict(time=np.array([0., .2, 99.]), position=np.array([[[0,0,2]], [[.2,0,2]], [[999,999,999]]]))
    changed = History(changed.time, changed.points, changed.bad, {'head': head})
    pc = history_sequence(changed, changed.centres[-1], steps=6, step_s=.1)
    np.testing.assert_array_equal(pa['values'], pc['values'])


def inputs(n=3, length=5):
    torch.manual_seed(63)
    x = torch.randn(n, length, 9)
    mask = torch.rand(n, length, 9) > .3
    mask[:, 0] = True
    elapsed = torch.full_like(x, .1)
    gap = torch.full((n,), .2)
    features = torch.randn(n, 4)
    prior = torch.randn(n, 3)
    age = torch.full((n,), .2)
    return x, mask, elapsed, gap, features, prior, age


def activate_decoder(model):
    with torch.no_grad():
        model.decoder[-1].weight.normal_(0, .1)


def test_decay_gru_ignores_unobserved_placeholders_and_responds_to_time():
    model = DecayGRU(4, hidden=8)
    activate_decoder(model)
    args = inputs()
    first = model(*args)
    changed = list(args)
    changed[0] = torch.where(args[1], args[0], torch.full_like(args[0], 999.))
    torch.testing.assert_close(first, model(*changed))
    changed = list(args)
    changed[3] = args[3]+.7
    assert not torch.allclose(first, model(*changed))
    first.square().mean().backward()
    assert torch.isfinite(model.input_decay_weight.grad).all()
    assert model.input_decay_weight.grad.abs().sum() > 0


def test_zero_age_has_exact_prior_for_directly_observed_state():
    model = DecayGRU(4, hidden=8)
    activate_decoder(model)
    args = list(inputs())
    args[-1] = torch.zeros_like(args[-1])
    torch.testing.assert_close(model(*args), args[-2])


def test_author_cru_masking_causal_forecast_and_finite_gradient():
    model = AuthorCRU(4, hidden=8)
    activate_decoder(model)
    args = inputs(n=2, length=3)
    first = model(*args)
    changed = list(args)
    changed[0] = torch.where(args[1], args[0], torch.full_like(args[0], 999.))
    torch.testing.assert_close(first, model(*changed))
    assert first.shape == (2, 3) and torch.isfinite(first).all()
    first.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_author_fast_cru_initialization_and_zero_time_are_finite():
    model = AuthorCRU(4, hidden=8, fast=True)
    args = list(inputs(n=2, length=3))
    args[3] = torch.zeros_like(args[3])
    assert torch.isfinite(model(*args)).all()


@pytest.mark.parametrize('kind', ['decay_gru', 'cru'])
def test_actual_event_intervals_affect_forecast_even_with_fresh_measurements(kind):
    torch.manual_seed(33)
    model = DecayGRU(4, hidden=8) if kind == 'decay_gru' else AuthorCRU(4, hidden=8)
    activate_decoder(model)
    args = list(inputs(n=2, length=3))
    args[1] = torch.ones_like(args[1])
    args[2] = torch.zeros_like(args[2])
    regular = torch.tensor([[0., .03, .03], [0., .03, .03]])
    irregular = torch.tensor([[0., .4, .7], [0., .4, .7]])
    assert torch.max(torch.abs(model(*args, regular)-model(*args, irregular))) > 1e-8


def test_sequence_gate_stays_inside_expert_coordinate_bounds_and_hides_placeholders():
    model = SequenceExpertGate(4, hidden=8)
    with torch.no_grad():
        model.gate[-1].weight.normal_(0, .1)
    args = inputs(n=2)
    experts = torch.randn(2, 3, 16)
    a = model(*args, experts=experts)
    assert (a >= experts.min(-1).values-1e-7).all()
    assert (a <= experts.max(-1).values+1e-7).all()
    changed = list(args)
    changed[0] = torch.where(args[1], args[0], torch.full_like(args[0], -999.))
    torch.testing.assert_close(a, model(*changed, experts=experts))


def test_replay_adapter_matches_manual_cpu_forward():
    from shuwei_c.q2_features import CausalExperts, transform_features
    h = history()
    now = .8
    causal = CausalExperts('EgoDemo').at(h, now)
    n = len(causal['feature_names'])
    model = DecayGRU(n, hidden=8)
    activate_decoder(model)
    state = dict(kind='decay_gru', hidden=8, observed_mean=np.zeros(9),
                 state_dict=model.state_dict(), feature_names=causal['feature_names'],
                 feature_mean=np.zeros(n), feature_std=np.ones(n), noise={'EgoDemo':{'hand_scale':.1}})
    predictor = SequencePredictor(state, 'EgoDemo')
    actual = predictor(h, now)
    seq = history_sequence(h, causal['anchor'])
    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32)
    x = transform_features(causal['features'], causal['feature_names'], [.1, .1]).clip(-10, 10)
    expected = model(tensor(seq['values']/.1), torch.as_tensor(seq['mask']), tensor(seq['elapsed']),
                     tensor([.3, .3]), tensor(x), tensor(causal['experts'][:,:,8]/.1),
                     tensor(causal['age']), tensor(seq['step_dt'])).detach().numpy()*.1+causal['anchor']
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    with pytest.raises(ValueError, match='precedes'):
        predictor(h, .4)
