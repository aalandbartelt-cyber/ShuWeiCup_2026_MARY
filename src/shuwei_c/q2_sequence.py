"""Causal masked-time sequence encoders for the Q2 sparse-observation study.

The decay-GRU is a regression adaptation inspired by GRU-D: explicit new-value
masks, per-feature ages, input decay to fitting-fold means, and hidden-state
decay. It is not a renamed ordinary GRU or an exact reproduction of GRU-D's
clinical classification task. CRU imports the unmodified authors' CRUCell from
the pinned, locally retained AGPL-3.0 repository; the wrapper/decoder is local.
AI-assisted: OpenAI Codex, 2026-11-20; source and adaptations documented.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib
import sys
import numpy as np
import torch

STEPS = 32
STEP_S = 1/30
SEQUENCE_FIELDS = ['own_x', 'own_y', 'own_z', 'other_x', 'other_y', 'other_z',
                   'head_x', 'head_y', 'head_z']


def history_sequence(history, anchor, steps=STEPS, step_s=STEP_S):
    """Only disclosed values at their actual source times, left-padded if needed.

    The event sequence never interpolates coordinates: values are most recent qualified
    disclosed samples. A mask is 1 only when that source sample is new to the
    sequence. Returned ages retain the actual source time rather than grid time.
    The forecast gap after this grid is handled separately in the network.
    """
    if steps < 2 or step_s <= 0:
        raise ValueError('Invalid sequence grid')
    t = np.asarray(history.time, float)
    if len(t) == 0 or np.any(np.diff(t) <= 0):
        raise ValueError('Nonempty increasing received source times required')
    centre = history.centres
    anchor = np.asarray(anchor, float)
    if anchor.shape != (2, 3):
        raise ValueError('Anchor must have one position per hand')
    selected = t[-steps:]
    grid = np.r_[selected[0]-np.arange(steps-len(selected), 0, -1)*step_s, selected]
    values = np.zeros((2, steps, 9), np.float32)
    masks = np.zeros_like(values, bool)
    ages = np.zeros_like(values)
    available = []
    for hand in range(2):
        good = ~history.bad[:, hand] & np.isfinite(centre[:, hand]).all(axis=-1)
        available.append((t[good], centre[good, hand]))
    head = history.auxiliary.get('head')
    if head is None:
        head_time, head_position = np.empty(0), np.empty((0, 3))
    else:
        head_time = np.asarray(head['time'], float)
        head_position = np.asarray(head['position'], float).reshape(-1, 3)
        finite = np.isfinite(head_position).all(axis=1)
        head_time, head_position = head_time[finite], head_position[finite]
    for hand in range(2):
        for channel, (times, positions) in enumerate([available[hand], available[1-hand], (head_time, head_position)]):
            indices = np.searchsorted(times, grid+1e-9, side='right')-1
            known = indices >= 0
            columns = slice(channel*3, channel*3+3)
            if known.any():
                values[hand, known, columns] = positions[indices[known]]-anchor[hand]
                ages[hand, known, columns] = np.maximum(grid[known]-times[indices[known]], 0)[:, None]
            fresh = known & (indices != np.r_[-1, indices[:-1]])
            masks[hand, fresh, columns] = True
    return dict(values=values, mask=masks, elapsed=ages,
                step_dt=np.tile(np.r_[0., np.diff(grid)], (2, 1)).astype(np.float32),
                grid_time=grid, last_source_time=float(t[-1]))


class DecayGRU(torch.nn.Module):
    def __init__(self, features, hidden=32, decay=True, mean=None):
        super().__init__()
        self.hidden, self.decay = hidden, decay
        self.register_buffer('observed_mean', torch.zeros(9) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.input_decay_weight = torch.nn.Parameter(torch.full((9,), .05))
        self.input_decay_bias = torch.nn.Parameter(torch.zeros(9))
        self.hidden_decay = torch.nn.Linear(9, hidden)
        torch.nn.init.constant_(self.hidden_decay.weight, .05)
        torch.nn.init.zeros_(self.hidden_decay.bias)
        self.cell = torch.nn.GRUCell(27, hidden)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden+features, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 3))
        torch.nn.init.zeros_(self.decoder[-1].weight)
        torch.nn.init.zeros_(self.decoder[-1].bias)

    def encode(self, values, mask, elapsed, gap, step_dt=None):
        h = values.new_zeros((len(values), self.hidden))
        previous = self.observed_mean[None].expand(len(values), -1)
        for k in range(values.shape[1]):
            m = mask[:, k].to(values.dtype)
            delta = elapsed[:, k]
            # Time since the previous qualified value, measured before this
            # event's update; fresh observations must not erase elapsed time.
            transition_delta = delta if k == 0 else elapsed[:, k-1]+(STEP_S if step_dt is None else step_dt[:, k, None])
            if self.decay:
                gamma_x = torch.exp(-torch.relu(delta*self.input_decay_weight+self.input_decay_bias))
                imputed = gamma_x*previous+(1-gamma_x)*self.observed_mean
                h = h*torch.exp(-torch.relu(self.hidden_decay(transition_delta)))
            else:
                imputed = previous
            x = m*values[:, k]+(1-m)*imputed
            h = self.cell(torch.cat([x, m, torch.log1p(transition_delta)], dim=-1), h)
            previous = m*values[:, k]+(1-m)*previous
        # The unobserved forecasting interval is one explicit missing step.
        delta = elapsed[:, -1]+gap[:, None]
        if self.decay:
            gamma_x = torch.exp(-torch.relu(delta*self.input_decay_weight+self.input_decay_bias))
            x = gamma_x*previous+(1-gamma_x)*self.observed_mean
            decayed = h*torch.exp(-torch.relu(self.hidden_decay(delta)))
        else:
            x, decayed = previous, h
        forecast_h = self.cell(torch.cat([x, torch.zeros_like(x), torch.log1p(delta)], dim=-1), decayed)
        h = torch.where((gap > 1e-9)[:, None], forecast_h, h)
        return h

    def forward(self, values, mask, elapsed, gap, features, prior, age, step_dt=None):
        h = self.encode(values, mask, elapsed, gap, step_dt)
        correction = self.decoder(torch.cat([h, features], dim=-1))
        return prior+(age/(age+.1))[:, None]*correction


class SequenceExpertGate(torch.nn.Module):
    """Use a missing-time encoder to choose a convex mixture of motion experts."""
    def __init__(self, features, hidden=32, mean=None, sequence=True):
        super().__init__()
        self.sequence = sequence
        self.encoder = DecayGRU(features, hidden, mean=mean) if sequence else None
        # The residual decoder is deliberately absent from this mechanism.
        if self.encoder is not None:
            self.encoder.decoder = torch.nn.Identity()
        self.gate = torch.nn.Sequential(torch.nn.Linear(features+(hidden if sequence else 0), hidden),
                                        torch.nn.SiLU(), torch.nn.Linear(hidden, 16))
        torch.nn.init.zeros_(self.gate[-1].weight)
        torch.nn.init.constant_(self.gate[-1].bias, -2.)
        with torch.no_grad():
            self.gate[-1].bias[8] = 2.

    def forward(self, values, mask, elapsed, gap, features, prior, age, step_dt=None, experts=None):
        if experts is None or experts.shape[1:] != (3, 16):
            raise ValueError('All 16 normalized numeric motion experts are required')
        representation = torch.cat([self.encoder.encode(values, mask, elapsed, gap, step_dt), features], 1) if self.sequence else features
        weights = torch.softmax(self.gate(representation), dim=-1)
        return torch.einsum('nck,nk->nc', experts, weights)


def load_author_cru(repo=None):
    root = Path(__file__).resolve().parents[2]
    repo = Path(repo) if repo else root/'artifacts/c_q2/sequence/external/Continuous-Recurrent-Units'
    vendor = repo.parent/'vendor'
    if not (repo/'lib/CRUCell.py').exists():
        raise FileNotFoundError('Pinned author CRU source is absent; see sequence provenance')
    for path in (vendor, repo):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module('lib.CRUCell').CRUCell


class AuthorCRU(torch.nn.Module):
    """Unmodified CRU cell with a masked numeric encoder and displacement head.

    This calls update/predict only in chronological source order. The final
    forecast has no measurement update. CRU full latent dimension is `hidden`.
    """
    def __init__(self, features, hidden=8, fast=False, repo=None):
        super().__init__()
        if hidden % 2 or hidden < 2:
            raise ValueError('CRU latent dimension must be even')
        self.hidden, self.lod, self.fast = hidden, hidden//2, fast
        args = SimpleNamespace(f_cru=fast, rkn=False, orthogonal=True,
                               num_basis=4, bandwidth=min(1, hidden//2-1),
                               t_sensitive_trans_net=False,
                               trans_net_hidden_units=[], trans_net_hidden_activation='Tanh',
                               trans_var_activation='exp', trans_covar=.1)
        self.cell = load_author_cru(repo)(self.lod, args, dtype=torch.float64)
        self.encoder = torch.nn.Sequential(torch.nn.Linear(27, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2*self.lod)).double()
        self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden+features, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3)).double()
        torch.nn.init.zeros_(self.decoder[-1].weight)
        torch.nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, values, mask, elapsed, gap, features, prior, age, step_dt=None):
        values, elapsed, gap = values.double(), elapsed.double(), gap.double()
        mask = mask.bool()
        # Hidden placeholders cannot enter the encoder through a partially
        # observed vector. Carry only values actually marked newly observed.
        previous = torch.zeros_like(values[:, 0])
        safe = []
        for k in range(values.shape[1]):
            previous = torch.where(mask[:, k], values[:, k], previous)
            safe.append(previous)
        safe = torch.stack(safe, dim=1)
        encoded = self.encoder(torch.cat([safe, mask.double(), torch.log1p(elapsed)], dim=-1))
        latent = torch.nn.functional.normalize(encoded[..., :self.lod], dim=-1, eps=1e-8)
        variance = torch.nn.functional.softplus(encoded[..., self.lod:])+1e-4
        mean = values.new_zeros((len(values), self.hidden))
        cov = [values.new_ones((len(values), self.lod)), values.new_ones((len(values), self.lod)), values.new_zeros((len(values), self.lod))]
        for k in range(values.shape[1]):
            if k:
                dt = values.new_full((len(values),), STEP_S) if step_dt is None else step_dt[:, k].double()
                mean, cov = self.cell._predict(mean, cov, dt)
            mean, cov, _ = self.cell._update(mean, cov, latent[:, k], variance[:, k], mask[:, k].any(dim=-1))
        mean, cov = self.cell._predict(mean, cov, gap)
        correction = self.decoder(torch.cat([mean, features.double()], dim=-1))
        return prior.double()+(age.double()/(age.double()+.1))[:, None]*correction


class SequencePredictor:
    """Replay adapter using only ``(History, now)`` and a declared dataset.

    The checkpoint must be locally trusted: PyTorch checkpoints contain Python
    normalization metadata. No record identifier, reference or path is read.
    """
    def __init__(self, checkpoint, dataset, device='cpu'):
        from .q2_features import CausalExperts
        state = torch.load(checkpoint, map_location='cpu', weights_only=False) if isinstance(checkpoint, (str, Path)) else checkpoint
        self.state, self.device = state, device
        self.extractor = CausalExperts(dataset)
        self.scale = float(state['noise'][dataset]['hand_scale'])
        kind = state['kind']
        self.model = (SequenceExpertGate(len(state['feature_names']), hidden=state['hidden'],
                                        mean=state['observed_mean'], sequence=kind == 'decay_gru_gate')
                      if kind in ('decay_gru_gate', 'static_gate') else
                      DecayGRU(len(state['feature_names']), hidden=state['hidden'],
                              decay=kind == 'decay_gru', mean=state['observed_mean'])
                      if kind.endswith('gru') else AuthorCRU(len(state['feature_names']),
                                                            hidden=state['hidden'], fast=kind == 'fcru'))
        self.model.load_state_dict(state['state_dict'])
        self.model.to(device).eval()
        self._history = None

    def __call__(self, history, now):
        from .q2_features import transform_features
        if now < history.time[-1]-1e-9:
            raise ValueError('Query precedes available source time')
        causal = self.extractor.at(history, now)
        if causal['feature_names'] != self.state['feature_names']:
            raise ValueError('Checkpoint feature schema differs')
        if history is not self._history:
            self._history = history
            self.sequence = history_sequence(history, causal['anchor'])
        sequence = self.sequence
        feature = transform_features(causal['features'], causal['feature_names'], np.full(2, self.scale, np.float32))
        feature = np.clip((feature-self.state['feature_mean'])/self.state['feature_std'], -10, 10)
        def tensor(value, dtype=torch.float32):
            return torch.as_tensor(value, device=self.device, dtype=dtype)
        gap = np.full(2, max(0., now-history.time[-1]), np.float32)
        with torch.inference_mode():
            inputs = [tensor(sequence['values']/np.float32(self.scale)),
                                      tensor(sequence['mask'], torch.bool), tensor(sequence['elapsed']),
                                      tensor(gap), tensor(feature),
                                      tensor(causal['experts'][:, :, 8]/np.float32(self.scale)),
                                      tensor(causal['age']), tensor(sequence['step_dt'])]
            if self.state['kind'] in ('decay_gru_gate', 'static_gate'):
                inputs.append(tensor(causal['experts']/np.float32(self.scale)))
            displacement = self.model(*inputs)
        return causal['anchor']+displacement.cpu().numpy()*self.scale
