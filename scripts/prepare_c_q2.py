#!/usr/bin/env python3
"""Audit Q2 development windows, frozen-Q1 hashes, folds and runtime resources."""
from pathlib import Path
import hashlib
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset
from shuwei_c.motion import estimate_noise
from shuwei_c.q2_replay import select_windows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_folds(records, seed, n=4):
    # Balance number of environments rather than records. Greedily balance the
    # per-dataset recording counts under a hard n_env / n fold-capacity limit.
    datasets = sorted({r.dataset for r in records})
    environments = sorted({r.environment for r in records})
    counts = {e: np.array([sum(r.environment == e and r.dataset == d for r in records)
                          for d in datasets], float) for e in environments}
    rng = np.random.default_rng(seed)
    tie = dict(zip(environments, rng.random(len(environments))))
    order = sorted(environments, key=lambda e: (-max(counts[e]), -sum(counts[e]), tie[e]))
    bins, sizes = [[] for _ in range(n)], np.zeros((n, len(datasets)))
    target = sum(counts.values()) / n
    for e in order:
        options = [i for i in range(n) if len(bins[i]) < int(np.ceil(len(environments)/n))]
        chosen = min(options, key=lambda i: (np.sum(((sizes[i]+counts[e])/np.maximum(target, 1))**2), len(bins[i]), i))
        bins[chosen].append(e); sizes[chosen] += counts[e]
    result = []
    for fold, held in enumerate(bins):
        fitting = [r for r in records if r.environment not in held]
        if set(r.dataset for r in fitting) != set(datasets):
            raise ValueError('A fold loses an entire dataset in training')
        result.append(dict(fold=fold, training_environments=sorted(set(environments)-set(held)),
                           held_environments=sorted(held), noise=estimate_noise(fitting)))
    return result


def main():
    started = time.perf_counter()
    out = Path('artifacts/c_q2/protocol'); out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path('configs/c_q2.json').read_text())
    data = Dataset()
    records = data.records('train')+data.records('validation')
    rows, eligible = [], []
    for r in records:
        windows, audit = select_windows(r, config)
        rows.extend(audit)
        print(json.dumps(dict(split=r.split, dataset=r.dataset, windows=len(windows))), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'window_audit.csv', index=False)
    frame[frame.reason == 'included'].to_csv(out/'development_windows.csv', index=False)
    frame.groupby(['split', 'dataset', 'horizon_s', 'reason'], observed=True).agg(
        windows=('window_id', 'size'), target_times=('target_times', 'sum')).reset_index().to_csv(out/'window_counts.csv', index=False)
    folds = environment_folds(data.records('train'), config['seed'])
    (out/'folds.json').write_text(json.dumps(folds, indent=2)+'\n')
    (out/'training_noise.json').write_text(json.dumps(estimate_noise(data.records('train')), indent=2)+'\n')
    frozen = {}
    for path in [Path('artifacts/c_q1_v3/frozen_numeric/freeze.json'),
                 Path('artifacts/c_q1_v3/frozen_quality/freeze.json')]:
        content = json.loads(path.read_text())
        frozen[str(path)] = digest(path)
        for key in ('source_hashes', 'inherited_v2_hashes'):
            for source, expected in content.get(key, {}).items():
                actual = digest(source)
                if actual != expected:
                    raise ValueError(f'Q1 frozen input changed: {source}')
                frozen[source] = actual
    (out/'q1_protected_hashes.json').write_text(json.dumps(frozen, indent=2)+'\n')
    import torch, lightgbm, scipy
    env = dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
               scipy=scipy.__version__, torch=torch.__version__, lightgbm=lightgbm.__version__,
               cuda=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    (out/'runtime.json').write_text(json.dumps(env, indent=2)+'\n')
    sources = ['configs/c_q2.json', __file__, 'src/shuwei_c/q2_replay.py',
               'src/shuwei_c/data.py', 'src/shuwei_c/motion.py']
    manifest = dict(test_used=False, elapsed_s=time.perf_counter()-started,
                    records=len(records), environments=len(set(r.environment for r in records)),
                    selected_windows=int((frame.reason == 'included').sum()),
                    source_hashes={str(p): digest(p) for p in sources})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
