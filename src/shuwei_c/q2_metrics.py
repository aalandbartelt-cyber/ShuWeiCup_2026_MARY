"""Explicit hierarchical aggregation; no frame-count dominance across records."""
import numpy as np
import pandas as pd


def summarize_windows(frame, group_columns):
    values = [c for c in ('mean_error', 'normalized_error', 'unobserved_error',
                         'unobserved_normalized_error', 'p95_error', 'max_error',
                         'requests', 'total_cost', 'waiting_union_s', 'latency_sum_s',
                         'responses_within_window', 'current_direct_fraction') if c in frame]
    output = []
    for key, g in frame.groupby(group_columns, dropna=False, sort=True, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        for dataset in ['ALL']+sorted(g.dataset.unique()):
            f = g if dataset == 'ALL' else g[g.dataset == dataset]
            by_record = f.groupby(['environment', 'episode'], observed=True)[values].mean()
            by_environment = by_record.groupby('environment', observed=True).mean()
            row = dict(zip(group_columns, key))
            row.update(dataset=dataset, windows=len(f), records=f.episode.nunique(),
                       environments=f.environment.nunique())
            row.update(by_environment.mean().to_dict())
            # Quantiles are means of within-window quantiles, not pooled tails.
            output.append(row)
    return pd.DataFrame(output)


def prediction_metrics(predictions, reference, direct, scale):
    errors = np.linalg.norm(predictions-reference, axis=-1)
    unobserved = errors[~np.asarray(direct, bool)]
    return dict(mean_error=float(errors.mean()), normalized_error=float(errors.mean()/scale),
                unobserved_error=float(unobserved.mean()) if unobserved.size else np.nan,
                unobserved_normalized_error=float(unobserved.mean()/scale) if unobserved.size else np.nan,
                p95_error=float(np.quantile(errors, .95)), max_error=float(errors.max()),
                current_direct_fraction=float(np.mean(direct)))
