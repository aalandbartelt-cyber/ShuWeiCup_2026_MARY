#!/usr/bin/env python3
"""Nested environment selection and OOF sparse motion predictions for Q2.

Only training/validation records are loaded. Main-horizon inner loss chooses
epochs; long-horizon samples remain in fitting and are separately stress-tested.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.data import Dataset
from shuwei_c.motion import estimate_noise
from shuwei_c.q2_learning import load_pack, subset, fit_gate, predict_gate, score_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('artifacts/c_q2/learning_data'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/c_q2/gate_oof'))
    parser.add_argument('--kinds', nargs='+', default=['mixture', 'residual'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[20261120,20261121,20261122])
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'manifest.json').exists():
        raise ValueError('Completed stage; choose a new output directory')
    train = load_pack(args.data/'train'); val = load_pack(args.data/'validation')
    folds_path = Path('artifacts/c_q2/protocol/folds.json')
    folds = json.loads(folds_path.read_text())
    records = Dataset().records('train'); full_noise = estimate_noise(records)
    files = [Path(__file__), Path('src/shuwei_c/q2_learning.py'), Path('src/shuwei_c/q2_features.py'),
             Path('src/shuwei_c/data.py'), Path('src/shuwei_c/motion.py'), folds_path,
             args.data/'train/manifest.json', args.data/'validation/manifest.json']
    signature = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    identity = dict(kinds=args.kinds, seeds=args.seeds, epochs=args.epochs,
                    input_hashes=signature, selection='nested environment, main h<=1 loss', test_used=False)
    ip = args.out/'identity.json'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Input or implementation differs from existing OOF stage')
    ip.write_text(json.dumps(identity, indent=2)+'\n')
    for p in files:
        relative = p.relative_to(Path.cwd()) if p.is_absolute() else p
        destination = args.out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, destination)
    torch.cuda.set_per_process_memory_fraction(.45)
    started = time.perf_counter(); all_summaries = []
    for kind in args.kinds:
        inner_epochs = {seed: [] for seed in args.seeds}
        for fold in folds:
            number = fold['fold']; root = args.out/f'{kind}/fold{number}'; root.mkdir(parents=True, exist_ok=True)
            marker = root/'complete.json'
            if marker.exists():
                saved = json.loads(marker.read_text())
                for seed in args.seeds:
                    inner_epochs[seed].append(saved['selected_epochs'][str(seed)])
                all_summaries.append(pd.read_csv(root/'summary.csv')); continue
            fit_mask = train['meta'].environment.isin(fold['training_environments']).to_numpy()
            held_mask = train['meta'].environment.isin(fold['held_environments']).to_numpy()
            inner_envs = folds[(number+1)%len(folds)]['held_environments']
            inner_mask = fit_mask & ~train['meta'].environment.isin(inner_envs).to_numpy()
            inner_held_mask = fit_mask & train['meta'].environment.isin(inner_envs).to_numpy() & (train['meta'].horizon_s<=1.).to_numpy()
            inner_noise = estimate_noise([r for r in records if r.environment in set(train['meta'].loc[inner_mask, 'environment'])])
            fitting, held = subset(train, fit_mask), subset(train, held_mask)
            inner_fit, inner_held = subset(train, inner_mask), subset(train, inner_held_mask)
            predictions, selected_epochs = [], {}
            for seed in args.seeds:
                checkpoint = root/f'seed{seed}.pt'
                selection_path = root/f'seed{seed}_selection.json'
                if checkpoint.exists() and selection_path.exists():
                    model = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    selection = json.loads(selection_path.read_text())
                else:
                    selected = fit_gate(inner_fit, inner_noise, inner_held, kind=kind, seed=seed, epochs=args.epochs)
                    model = fit_gate(fitting, fold['noise'], kind=kind, seed=seed, epochs=selected['epochs'])
                    torch.save(model, checkpoint)
                    selection = dict(epochs=selected['epochs'], trace=selected['trace'],
                                     fit_environments=sorted(set(train['meta'].loc[inner_mask,'environment'])),
                                     inner_held_environments=inner_envs, noise=inner_noise)
                    selection_path.write_text(json.dumps(selection, indent=2)+'\n')
                selected_epochs[str(seed)] = selection['epochs']; inner_epochs[seed].append(selection['epochs'])
                predictions.append(predict_gate(held, model))
            prediction = np.mean(predictions, axis=0)
            frame, summary = score_predictions(held, prediction, fold['noise'])
            frame['fold'] = number; frame['method'] = kind; frame['split'] = 'train_oof'
            frame['scale'] = frame.dataset.map({d:v['hand_scale'] for d,v in fold['noise'].items()})
            for k, coordinate in enumerate('xyz'):
                frame['prediction_'+coordinate] = prediction[:,k]+held['anchor'][:,k]
            frame.to_parquet(root/'predictions.parquet', index=False)
            summary['fold'] = number; summary['method'] = kind; summary['split'] = 'train_oof'
            summary.to_csv(root/'summary.csv', index=False); all_summaries.append(summary)
            result = dict(fold=number, kind=kind, selected_epochs=selected_epochs,
                          training_environments=fold['training_environments'], held_environments=fold['held_environments'],
                          rows=len(frame), elapsed_s=time.perf_counter()-started)
            marker.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result), flush=True)
            del fitting, held, inner_fit, inner_held, predictions
        root = args.out/f'{kind}/full'; root.mkdir(parents=True, exist_ok=True)
        marker = root/'complete.json'
        if marker.exists():
            all_summaries.append(pd.read_csv(root/'summary.csv')); continue
        predictions, epochs = [], {}
        for seed in args.seeds:
            epochs[seed] = max(1, int(np.median(inner_epochs[seed])))
            model = fit_gate(train, full_noise, kind=kind, seed=seed, epochs=epochs[seed])
            torch.save(model, root/f'seed{seed}.pt'); predictions.append(predict_gate(val, model))
        prediction = np.mean(predictions, axis=0)
        frame, summary = score_predictions(val, prediction, full_noise)
        frame['fold'] = -1; frame['method'] = kind
        frame['scale'] = frame.dataset.map({d:v['hand_scale'] for d,v in full_noise.items()})
        for k, coordinate in enumerate('xyz'):
            frame['prediction_'+coordinate] = prediction[:,k]+val['anchor'][:,k]
        frame.to_parquet(root/'predictions.parquet', index=False)
        summary['fold'] = -1; summary['method'] = kind; summary['split'] = 'validation'
        summary.to_csv(root/'summary.csv', index=False); all_summaries.append(summary)
        marker.write_text(json.dumps(dict(selected_epochs=epochs, rows=len(frame), elapsed_s=time.perf_counter()-started), indent=2)+'\n')
    pd.concat(all_summaries, ignore_index=True).to_csv(args.out/'comparison.csv', index=False)
    (args.out/'manifest.json').write_text(json.dumps(dict(**identity, elapsed_s=time.perf_counter()-started), indent=2)+'\n')


if __name__ == '__main__':
    main()
