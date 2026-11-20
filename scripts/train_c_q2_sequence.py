#!/usr/bin/env python3
"""Training-environment held-out sequence probe on keys shared with numeric gates.

This script never opens the official test or validation records. Selection,
normalization and early stopping belong solely to one preassigned training fold.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.q2_learning import load_pack, subset, normalized, score_predictions
from shuwei_c.q2_sequence import DecayGRU, AuthorCRU, SequenceExpertGate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sequences(numeric, folder):
    """Read compact unique states and global row-to-state maps, checking joins."""
    manifest = json.loads((folder/'manifest.json').read_text())
    fields = {k: [] for k in ['values', 'mask', 'elapsed', 'step_dt', 'sequence_index', 'query_gap']}
    offset = 0
    for path in sorted(numeric.glob('*.parquet')):
        marker = json.loads((folder/f'{path.stem}.json').read_text())
        target = folder/f'{path.stem}.npz'
        if marker['metadata_sha256'] != sha(path) or marker['numeric_npz_sha256'] != sha(path.with_suffix('.npz')) or marker['sequence_sha256'] != sha(target):
            raise ValueError('Sequence join fingerprint mismatch')
        with np.load(target) as data:
            if len(data['sequence_index']) != marker['rows']:
                raise ValueError('Sequence join row count mismatch')
            for key in fields:
                fields[key].append(data[key]+offset if key == 'sequence_index' else data[key])
            offset += len(data['values'])
    result = {k: np.concatenate(v) for k, v in fields.items()}
    if len(result['sequence_index']) != manifest['rows']:
        raise ValueError('Incomplete sequence cache')
    return result


def sample_indices(meta, fitting, per_group, seed):
    """Fixed scenario-balanced samples, independent of labels and candidate method."""
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in meta.loc[fitting].groupby(['episode', 'horizon_s', 'scenario', 'hand'], sort=True):
        index = group.index.to_numpy()
        selected.extend(rng.choice(index, min(per_group, len(index)), replace=False))
    return np.array(sorted(selected), dtype=np.int64)


def tensor_pack(pack, sequences, row_index, noise, mean, std, device):
    selected = subset(pack, row_index)
    raw = normalized(selected, noise)
    state = sequences['sequence_index'][row_index]
    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(value, device=device, dtype=dtype)
    return dict(values=tensor(sequences['values'][state]/raw['scales'][:, None, None]),
                mask=tensor(sequences['mask'][state], torch.bool),
                elapsed=tensor(sequences['elapsed'][state]),
                step_dt=tensor(sequences['step_dt'][state]),
                gap=tensor(sequences['query_gap'][row_index]),
                features=tensor(np.clip((raw['x']-mean)/std, -10, 10)),
                prior=tensor(raw['e'][:, :, 8]), experts=tensor(raw['e']), age=tensor(selected['meta'].age_s.to_numpy()),
                target=tensor(raw['y']), weight=tensor(raw['weights']*len(row_index))), selected


def forward(model, data, ix):
    values = [data[k][ix] for k in ['values', 'mask', 'elapsed', 'gap', 'features', 'prior', 'age', 'step_dt']]
    if isinstance(model, SequenceExpertGate):
        values.append(data['experts'][ix])
    return model(*values)


def predict(model, data, batch):
    result = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(data['target']), batch):
            result.append(forward(model, data, slice(begin, begin+batch)).cpu().numpy())
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('artifacts/c_q2/learning_data'))
    parser.add_argument('--sequences', type=Path, default=Path('artifacts/c_q2/sequence/data'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/c_q2/sequence/probe_fold0'))
    parser.add_argument('--kinds', nargs='+', choices=['decay_gru', 'masked_gru', 'cru', 'fcru', 'decay_gru_gate', 'static_gate'], default=['decay_gru', 'cru'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--per-group', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261120)
    parser.add_argument('--sample-seed', type=int, default=20261120)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--memory-fraction', type=float, default=.2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'manifest.json').exists():
        raise ValueError('Completed probe exists; use a new output directory')
    torch.set_num_threads(2)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    pack = load_pack(args.data/'train')
    sequences = load_sequences(args.data/'train', args.sequences/'train')
    if len(sequences['sequence_index']) != len(pack['meta']):
        raise ValueError('Numeric and sequence rows disagree')
    folds_path = Path('artifacts/c_q2/protocol/folds.json')
    fold = json.loads(folds_path.read_text())[args.fold]
    fitting = pack['meta'].environment.isin(fold['training_environments']).to_numpy()
    held = pack['meta'].environment.isin(fold['held_environments']).to_numpy()
    if not np.all(fitting ^ held):
        raise ValueError('Fold environments do not partition training records')
    fit_ix = sample_indices(pack['meta'], fitting, args.per_group, args.sample_seed)
    held_ix = sample_indices(pack['meta'], held, args.per_group, args.sample_seed+1)
    np.savez(args.out/'sample_indices.npz', fitting=fit_ix, held=held_ix)
    selected = subset(pack, fit_ix)
    raw = normalized(selected, fold['noise'])
    mean = np.average(raw['x'], axis=0, weights=raw['weights'])
    std = np.maximum(np.sqrt(np.average((raw['x']-mean)**2, axis=0, weights=raw['weights'])), 1e-4)
    data_fit, fitting_pack = tensor_pack(pack, sequences, fit_ix, fold['noise'], mean, std, args.device)
    data_held, held_pack = tensor_pack(pack, sequences, held_ix, fold['noise'], mean, std, args.device)
    # Means use only actually observed feature coordinates in fitting samples.
    observed_weight = data_fit['mask'].float()*data_fit['weight'][:, None, None]
    observed_mean = ((data_fit['values']*observed_weight).sum((0, 1))/observed_weight.sum((0, 1)).clamp_min(1e-8)).cpu().numpy()
    sources = [Path(__file__), Path('src/shuwei_c/q2_sequence.py'), Path('src/shuwei_c/q2_learning.py')]
    external = Path('artifacts/c_q2/sequence/external/Continuous-Recurrent-Units')
    identity = dict(fold=args.fold, noise=fold['noise'], fitting_environments=fold['training_environments'],
                    held_environments=fold['held_environments'], fitting_rows=len(fit_ix), held_rows=len(held_ix),
                    source_hashes={str(p): sha(p) for p in sources}, folds_sha256=sha(folds_path),
                    numeric_manifest_sha256=sha(args.data/'train/manifest.json'),
                    sequence_manifest_sha256=sha(args.sequences/'train/manifest.json'),
                    sample_indices_sha256=sha(args.out/'sample_indices.npz'),
                    seed=args.seed, sample_seed=args.sample_seed, per_group=args.per_group, max_epochs=args.epochs, patience=args.patience,
                    batch_size=args.batch, kinds=args.kinds, optimizer='torch.optim.AdamW',
                    learning_rate=.001, weight_decay=.01, gradient_clip=2., objective='balanced mean Euclidean error',
                    device=args.device, hardware=torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU',
                    memory_fraction=args.memory_fraction if args.device == 'cuda' else None,
                    torch_version=torch.__version__, author_cru_commit='79723e7772102cae8a78109e18d81beff9c7cac2',
                    author_cru_cell_sha256=sha(external/'lib/CRUCell.py'),
                    author_code_modified=False, official_validation_used=False, test_used=False,
                    status='training-environment probe; early-stopping and reporting use same held fold')
    (args.out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    print(json.dumps(identity, indent=2), flush=True)
    baseline = data_held['prior'].cpu().numpy()
    target = data_held['target'].cpu().numpy()
    weights = data_held['weight'].cpu().numpy()/len(held_ix)
    scale = np.array([fold['noise'][d]['hand_scale'] for d in held_pack['meta'].dataset])
    baseline_score = float(np.dot(weights, np.linalg.norm(baseline-target, axis=1)))
    summaries = []
    rows, summary = score_predictions(held_pack, baseline*scale[:, None], fold['noise'])
    summary['method'] = 'damped_0.2'; summaries.append(summary)
    summary.to_csv(args.out/'damped_0.2_summary.csv', index=False)
    results = []
    for kind in args.kinds:
        torch.manual_seed(args.seed)
        model = (SequenceExpertGate(len(mean), hidden=32, mean=observed_mean, sequence=kind == 'decay_gru_gate')
                 if kind in ('decay_gru_gate', 'static_gate') else
                 DecayGRU(len(mean), hidden=32, decay=kind == 'decay_gru', mean=observed_mean)
                 if kind.endswith('gru') else AuthorCRU(len(mean), hidden=8, fast=kind == 'fcru'))
        model.to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        generator = torch.Generator(device=args.device); generator.manual_seed(args.seed)
        best, best_state, best_epoch, stale, trace = float('inf'), None, 0, 0, []
        started = time.perf_counter()
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        initial_prediction = predict(model, data_held, args.batch)
        initial_score = float(np.dot(weights, np.linalg.norm(initial_prediction-target, axis=1)))
        failure = None
        for epoch in range(1, args.epochs+1):
            model.train(); total = 0.
            for ix in torch.randperm(len(fit_ix), generator=generator, device=args.device).split(args.batch):
                prediction = forward(model, data_fit, ix)
                loss = (torch.sqrt((prediction-data_fit['target'][ix]).square().sum(1)+1e-8)*data_fit['weight'][ix]).mean()
                if not torch.isfinite(loss):
                    failure = f'Nonfinite objective at epoch {epoch}'
                    break
                optimizer.zero_grad(); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                if not torch.isfinite(norm):
                    failure = f'Nonfinite gradient at epoch {epoch}'
                    break
                optimizer.step()
                total += float(loss.detach())*len(ix)/len(fit_ix)
            if failure:
                break
            predicted = predict(model, data_held, args.batch)
            score = float(np.dot(weights, np.linalg.norm(predicted-target, axis=1)))
            row = dict(kind=kind, epoch=epoch, train_objective=total, inner_held_error=score,
                       damped_baseline_error=baseline_score, elapsed_s=time.perf_counter()-started)
            trace.append(row); print(json.dumps(row), flush=True)
            pd.DataFrame(trace).to_csv(args.out/f'{kind}_trace.csv', index=False)
            if score < best-1e-6:
                best, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= args.patience:
                break
        record = dict(method=kind, best_epoch=best_epoch, normalized_error=best,
                      initialized_normalized_error=initial_score,
                      baseline_normalized_error=baseline_score, relative_change=best/baseline_score-1,
                      elapsed_s=time.perf_counter()-started, failure=failure,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == 'cuda' else None)
        if best_state is not None:
            model.load_state_dict(best_state)
            predicted = predict(model, data_held, args.batch)*scale[:, None]
            rows, summary = score_predictions(held_pack, predicted, fold['noise'])
            for k, coordinate in enumerate('xyz'):
                rows['displacement_'+coordinate] = predicted[:, k]
            rows['method'] = kind
            rows.to_parquet(args.out/f'{kind}_inner_held.parquet', index=False)
            summary['method'] = kind; summaries.append(summary)
            summary.to_csv(args.out/f'{kind}_summary.csv', index=False)
            torch.save(dict(state_dict=best_state, kind=kind, hidden=8 if kind in ('cru','fcru') else 32,
                            observed_mean=observed_mean, feature_mean=mean, feature_std=std,
                            noise=fold['noise'], best_epoch=best_epoch, feature_names=pack['feature_names'],
                            expert_names=pack['expert_names'], identity=identity), args.out/f'{kind}.pt')
        (args.out/f'{kind}.json').write_text(json.dumps(record, indent=2)+'\n')
        results.append(record)
        del model, optimizer
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    pd.concat(summaries, ignore_index=True).to_csv(args.out/'comparison.csv', index=False)
    (args.out/'manifest.json').write_text(json.dumps(dict(**identity, results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
