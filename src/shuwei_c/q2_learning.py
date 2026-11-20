"""Fold-scaled, environment-balanced sparse motion gating on the local GPU."""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import torch
from .data import balanced_weights
from .q2_features import transform_features


def load_pack(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    arrays = {key: [] for key in ('features', 'experts', 'target', 'anchor')}
    frames = []
    for path in sorted(folder.glob('*.parquet')):
        frames.append(pd.read_parquet(path))
        with np.load(path.with_suffix('.npz')) as data:
            for key in arrays:
                arrays[key].append(data[key])
    result = {key: np.concatenate(value) for key, value in arrays.items()}
    result['meta'] = pd.concat(frames, ignore_index=True)
    result['feature_names'] = manifest['feature_names']
    result['expert_names'] = manifest['expert_names']
    if len(result['meta']) != manifest['rows'] or any(len(result[k]) != manifest['rows'] for k in arrays):
        raise ValueError('Cached row counts disagree')
    return result


def subset(pack, mask):
    result = {k: pack[k][mask] for k in ('features', 'experts', 'target', 'anchor')}
    result['meta'] = pack['meta'].loc[mask].reset_index(drop=True)
    result['feature_names'] = pack['feature_names']; result['expert_names'] = pack['expert_names']
    return result


def row_weights(meta):
    columns = ['environment', 'episode', 'window_id', 'scenario', 'hand']
    counts = meta.groupby(columns, observed=True, sort=False).size().rename('n').reset_index()
    counts['weight'] = balanced_weights(counts.environment, counts.episode)/counts.n
    return meta.merge(counts[columns+['weight']], on=columns, how='left', sort=False).weight.to_numpy()


def normalized(pack, noise):
    scales = np.array([noise[d]['hand_scale'] for d in pack['meta'].dataset], np.float32)
    return dict(x=transform_features(pack['features'], pack['feature_names'], scales),
                e=pack['experts']/scales[:, None, None], y=pack['target']/scales[:, None],
                scales=scales, weights=row_weights(pack['meta']))


class Gate(torch.nn.Module):
    def __init__(self, features, experts, width=48, kind='mixture', prior=8,
                 mean_age=0., std_age=1.):
        super().__init__()
        self.kind, self.prior = kind, prior
        if kind == 'age_mixture':
            self.register_buffer('age_mean', torch.tensor(float(mean_age)))
            self.register_buffer('age_std', torch.tensor(float(std_age)))
            self.register_buffer('age_knots', torch.log1p(torch.tensor([0., 1/30, .1, .2, .5, 1., 2.])))
            self.logits = torch.nn.Parameter(torch.full((7, experts), -2.))
            with torch.no_grad():
                self.logits[:, prior] = 2.
            return
        self.net = torch.nn.Sequential(torch.nn.Linear(features, width), torch.nn.SiLU(),
                                       torch.nn.Linear(width, width), torch.nn.SiLU(),
                                       torch.nn.Linear(width, experts))
        torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)
        if kind == 'mixture':
            with torch.no_grad():
                self.net[-1].bias.fill_(-2.)
                self.net[-1].bias[prior] = 2.

    def forward(self, x, experts):
        if self.kind == 'age_mixture':
            age = torch.clamp(x[:, 0]*self.age_std+self.age_mean,
                              self.age_knots[0], self.age_knots[-1])
            right = torch.searchsorted(self.age_knots, age.contiguous(), right=True).clamp(1, len(self.age_knots)-1)
            left = right-1
            fraction = ((age-self.age_knots[left])/(self.age_knots[right]-self.age_knots[left])).unsqueeze(1)
            raw = (1-fraction)*self.logits[left]+fraction*self.logits[right]
            weights = torch.softmax(raw, dim=-1)
            smooth = (torch.softmax(self.logits, dim=-1)[1:]-torch.softmax(self.logits, dim=-1)[:-1]).square().mean()
            return torch.einsum('nck,nk->nc', experts, weights), smooth
        raw = self.net(x)
        if self.kind == 'mixture':
            weights = torch.softmax(raw, dim=-1)
            penalty = ((weights-torch.nn.functional.one_hot(torch.full((len(x),), self.prior,
                         device=x.device), experts.shape[-1]))**2).mean()
        else:
            weights = .5*torch.tanh(raw)
            penalty = (weights**2).mean()
            weights = weights.clone(); weights[:, self.prior] += 1.
        return torch.einsum('nck,nk->nc', experts, weights), penalty


def fit_gate(train, noise, validation=None, width=48, kind='mixture',
             regularization=.001, seed=20261120, epochs=80, patience=10,
             batch_size=8192):
    torch.manual_seed(seed); torch.set_num_threads(3)
    raw = normalized(train, noise); w = raw['weights']
    mean = np.average(raw['x'], axis=0, weights=w)
    std = np.maximum(np.sqrt(np.average((raw['x']-mean)**2, axis=0, weights=w)), 1e-4)
    def tensors(pack):
        data = normalized(pack, noise)
        x = torch.tensor(np.clip((data['x']-mean)/std, -10, 10), device='cuda', dtype=torch.float32)
        return (x, torch.tensor(data['e'], device='cuda'), torch.tensor(data['y'], device='cuda'),
                torch.tensor(data['weights']*len(data['weights']), device='cuda', dtype=torch.float32))
    x, e, y, weights = tensors(train)
    held = tensors(validation) if validation is not None else None
    net = Gate(x.shape[1], e.shape[-1], width, kind,
               mean_age=mean[0], std_age=std[0]).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator(device='cuda'); generator.manual_seed(seed)
    best, best_state, best_epoch, stale, trace = float('inf'), None, 0, 0, []
    for epoch in range(1, epochs+1):
        net.train(); total = 0.
        for ix in torch.randperm(len(x), generator=generator, device='cuda').split(batch_size):
            pred, penalty = net(x[ix], e[ix])
            distance = torch.sqrt(((pred-y[ix])**2).sum(dim=1)+1e-8)
            loss = (distance*weights[ix]).mean()+regularization*penalty
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.); optimizer.step()
            total += float(loss.detach())*len(ix)/len(x)
        score = total
        if held is not None:
            net.eval(); score = 0.
            with torch.inference_mode():
                hx, he, hy, hw = held
                for start in range(0, len(hx), batch_size):
                    end = start+batch_size
                    pred, _ = net(hx[start:end], he[start:end])
                    score += float((torch.linalg.vector_norm(pred-hy[start:end], dim=1)*hw[start:end]).sum())/len(hx)
        trace.append(dict(epoch=epoch, train_objective=total, inner_validation_error=score))
        if held is None or score < best-1e-6:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if epoch % 10 == 0:
            print(json.dumps(dict(kind=kind, seed=seed, epoch=epoch, train=total, held=score)), flush=True)
        if held is not None and stale >= patience:
            break
    result = dict(state_dict=best_state, mean=mean, std=std, noise=noise,
                  width=width, kind=kind, regularization=regularization, seed=seed,
                  epochs=best_epoch, trace=trace, feature_names=train['feature_names'],
                  expert_names=train['expert_names'], optimizer='torch.optim.AdamW',
                  learning_rate=.001, weight_decay=.01, batch_size=batch_size,
                  stopping='inner-environment patience' if held is not None else 'fixed inner-selected epochs')
    del x, e, y, weights, held, net, optimizer
    torch.cuda.empty_cache()
    return result


def predict_gate(pack, model, batch_size=16384):
    data = normalized(pack, model['noise'])
    net = Gate(len(model['feature_names']), len(model['expert_names']), model['width'], model['kind']).cuda()
    net.load_state_dict(model['state_dict']); net.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(data['x']), batch_size):
            stop = start+batch_size
            x = torch.tensor(np.clip((data['x'][start:stop]-model['mean'])/model['std'], -10, 10), device='cuda', dtype=torch.float32)
            e = torch.tensor(data['e'][start:stop], device='cuda')
            p, _ = net(x, e)
            predictions.append(p.cpu().numpy()*data['scales'][start:stop, None])
    return np.concatenate(predictions)


def score_predictions(pack, displacement, noise):
    errors = np.linalg.norm(displacement-pack['target'], axis=1)
    scales = np.array([noise[d]['hand_scale'] for d in pack['meta'].dataset])
    f = pack['meta'].copy(); f['error'] = errors; f['normalized_error'] = errors/scales
    rows = []
    for (horizon, cap, delay), g in f.groupby(['horizon_s', 'cap', 'delay_s'], observed=True):
        for dataset in ['ALL']+list(sorted(g.dataset.unique())):
            part = g if dataset == 'ALL' else g[g.dataset == dataset]
            weights = row_weights(part)
            rows.append(dict(horizon_s=horizon, cap=cap, delay_s=delay, dataset=dataset,
                             mean_error=float(np.dot(weights, part.error)),
                             normalized_error=float(np.dot(weights, part.normalized_error)),
                             n=len(part), environments=part.environment.nunique()))
    return f, pd.DataFrame(rows)
