#!/usr/bin/env python3
"""Exploratory Q1 review; never overwrite v1 or read test trajectories.

AI-assisted by Codex, 2026-11-20. These diagnostics are development evidence,
not an independent final evaluation or physical-ground-truth measurement.
"""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score
from shuwei_c.data import Dataset, balanced_weights
from shuwei_c.q1 import contiguous_runs

OUT = Path('artifacts/c_q1_review')
BASE = Path('artifacts/c_q1_v1')


def main():
    OUT.mkdir(exist_ok=True)
    data = Dataset()
    model = json.loads((BASE / 'state_model.json').read_text())
    coef = np.array(json.loads((BASE / 'multiscale_model.json').read_text())['0.1s']['coefficients'])
    # One fitted scalar, identical training masks/scales/weights to multiscale v1.
    moments = []
    all_runs, reason_rows = [], []
    for split in ['train', 'validation']:
        for rec in data.records(split):
            t, p, bad = rec.hands()
            for start, stop in contiguous_runs(bad.any(axis=1)):
                all_runs.append(dict(split=split, dataset=rec.dataset, environment=rec.environment,
                    episode=rec.episode, duration=float(t[stop-1]-t[start]+np.median(np.diff(t))),
                    left_censored=start == 0, right_censored=stop == len(t)))
            reasons = rec.streams['quality']['reasons']
            if reasons.ndim == 1:
                normalized = np.full(bad.shape, '', dtype=object)
                for i, raw in enumerate(reasons):
                    parts = str(raw).split('|')
                    for j, side in enumerate(['left', 'right']):
                        assigned = [v.split(':', 1)[1] for v in parts if v.lower().startswith(side+':')]
                        normalized[i,j] = ';'.join(assigned) or ('unassigned_bad' if bad[i,j] else '')
                reasons = normalized
            for reason, count in zip(*np.unique(reasons, return_counts=True)):
                reason_rows.append(dict(split=split, dataset=rec.dataset, episode=rec.episode,
                                        reason=str(reason), hand_frames=int(count)))
    pd.DataFrame(all_runs).to_csv(OUT/'development_bad_runs.csv', index=False)
    pd.DataFrame(reason_rows).to_csv(OUT/'quality_reason_counts.csv', index=False)
    for rec in data.records('train'):
        t, p, bad = rec.hands(); c = p.mean(axis=2)
        i = np.arange(2, len(t)-1); j = np.searchsorted(t, t[i]+.1-1e-9)
        take = j < len(t); i, j = i[take], j[take]
        v = (c[i]-c[i-1])/(t[i]-t[i-1])[:, None, None]
        target = (c[j]-c[i])/(t[j]-t[i])[:, None, None]
        valid = ~bad[j] & ~bad[i]
        for lag in [1, 3, 6, 12]: valid &= ~bad[np.maximum(i-lag, 0)]
        scale = model['base_training_noise'][rec.dataset]['hand_scale']
        x, y = v[valid].ravel()/scale, target[valid].ravel()/scale
        moments.append(dict(environment=rec.environment, episode=rec.episode,
                            xx=float(np.mean(x*x)), xy=float(np.mean(x*y))))
    m = pd.DataFrame(moments); w = balanced_weights(m.environment, m.episode)
    shrinkage = float(np.dot(w, m.xy)/np.dot(w, m.xx))
    (OUT/'scalar_velocity_model.json').write_text(json.dumps(dict(
        coefficient=shrinkage, fit='training only; one scalar; no validation tuning',
        mask='same qualified hand/frame set as multiscale training'), indent=2)+'\n')
    state_rows, coverage_rows, event_rows, horizon_rows = [], [], [], []
    for rec in data.records('validation'):
        t, p, bad = rec.hands(); c = p.mean(axis=2)
        i = np.arange(2, len(t)-1); j = np.searchsorted(t, t[i]+.1-1e-9)
        take = j < len(t); i, j = i[take], j[take]; dt = t[j]-t[i]
        v = (c[i]-c[i-1])/(t[i]-t[i-1])[:, None, None]
        x = np.stack([(c[i]-c[np.maximum(i-lag, 0)])/(t[i]-t[np.maximum(i-lag, 0)])[:, None, None]
                      for lag in [1, 3, 6, 12]], axis=-1)
        predictions = dict(linear=c[i]+v*dt[:, None, None],
                           scalar_damped=c[i]+shrinkage*v*dt[:, None, None],
                           multiscale_linear=c[i]+np.einsum('nhck,k->nhc', x, coef)*dt[:, None, None])
        z = np.load(BASE/'states'/f'{rec.episode}.npz')
        # Shared isotropic xyz covariance, with reference observation noise included.
        np.testing.assert_array_equal(z['time'], t)
        np.testing.assert_allclose(z['observed_centroid'], c, rtol=1e-12, atol=1e-14)
        cov = z['covariance'][i]
        noise = model['selected_noise'][rec.dataset]
        var = cov[:, :, 0, 0]+2*dt[:, None]*cov[:, :, 0, 1]+dt[:, None]**2*cov[:, :, 1, 1]
        var += noise['acceleration']*dt[:, None]**4/4+noise['r']
        assert np.isfinite(var).all() and np.all(var > 0)
        filtered = z['centroid'][i]+z['velocity'][i]*dt[:, None, None]
        for hand in range(2):
            valid = ~bad[j, hand]
            for method, pred in predictions.items():
                error = np.linalg.norm(pred[:, hand]-c[j, hand], axis=-1)
                for context in ['all', 'recent_flag', 'unflagged_history']:
                    recent = bad[i, hand] | bad[i-1, hand]
                    use = valid & (recent if context == 'recent_flag' else ~recent if context == 'unflagged_history' else True)
                    if not use.any(): continue
                    state_rows.append(dict(dataset=rec.dataset, environment=rec.environment,
                        episode=rec.episode, hand=hand, method=method, context=context, n=int(use.sum()),
                        error=float(error[use].mean()), p90=float(np.quantile(error[use], .9))))
            sq = np.sum((filtered[:, hand]-c[j, hand])**2, axis=-1)/var[:, hand]
            for level in [.5, .8, .9, .95]:
                limit = chi2.ppf(level, 3)
                coverage_rows.append(dict(dataset=rec.dataset, environment=rec.environment,
                    episode=rec.episode, hand=hand, nominal=level, n=int(valid.sum()),
                    coverage=float(np.mean(sq[valid] <= limit)),
                    radius=float(np.mean(np.sqrt(var[valid, hand]*limit)))))
        horizon_rows.append(dict(dataset=rec.dataset, environment=rec.environment, episode=rec.episode,
            n=len(i), minimum=float(dt.min()), median=float(np.median(dt)), maximum=float(dt.max()),
            over_110ms=int((dt > .110).sum()), over_125ms=int((dt > .125).sum())))
        for start, stop in contiguous_runs(bad.any(axis=1)):
            event_rows.append(dict(dataset=rec.dataset, environment=rec.environment, episode=rec.episode,
                start=float(t[start]), duration=float(t[stop-1]-t[start]+np.median(np.diff(t))),
                left_censored=start == 0, right_censored=stop == len(t)))
    states = pd.DataFrame(state_rows); states.to_csv(OUT/'state_comparison_by_record.csv', index=False)
    summary = states.groupby(['dataset','method','context','environment','episode']).error.mean().groupby(
        ['dataset','method','context','environment']).mean().groupby(['dataset','method','context']).mean().reset_index()
    summary.to_csv(OUT/'state_comparison_summary.csv', index=False)
    pivot = summary[summary.context=='all'].pivot(index='dataset', columns='method', values='error')
    old = pd.read_csv(BASE/'state_main_table.csv').set_index('dataset').loc[pivot.index]
    for name in ['linear', 'multiscale_linear']:
        np.testing.assert_allclose(pivot[name], old[name], rtol=1e-12, atol=1e-14)
    pivot['multiscale_gain_vs_scalar_pct'] = 100*(1-pivot.multiscale_linear/pivot.scalar_damped)
    pivot['scalar_share_of_gain_pct'] = 100*(pivot.linear-pivot.scalar_damped)/(pivot.linear-pivot.multiscale_linear)
    pivot.to_csv(OUT/'state_incremental_gain.csv')
    record_gain = states[states.context=='all'].groupby(['dataset','environment','episode','method']).error.mean().unstack()
    record_gain['multiscale_gain_pct'] = 100*(1-record_gain.multiscale_linear/record_gain.linear)
    record_gain['incremental_gain_pct'] = 100*(1-record_gain.multiscale_linear/record_gain.scalar_damped)
    record_gain.to_csv(OUT/'state_record_gain.csv')
    coverage = pd.DataFrame(coverage_rows); coverage.to_csv(OUT/'raw_filter_coverage_by_record.csv', index=False)
    coverage_summary = coverage.groupby(['dataset','nominal','environment','episode'])[['coverage','radius']].mean().groupby(
        ['dataset','nominal','environment']).mean().groupby(['dataset','nominal']).mean().reset_index()
    coverage_summary.to_csv(OUT/'raw_filter_coverage_summary.csv', index=False)
    pd.DataFrame(event_rows).to_csv(OUT/'validation_bad_runs.csv', index=False)
    pd.DataFrame(horizon_rows).to_csv(OUT/'horizon_diagnostics.csv', index=False)
    # Separate new onset from persistence; conditional weights are renormalized.
    pred = pd.read_csv(BASE/'quality_validation_predictions.csv'); quality_rows=[]
    pred['always_good'] = 0.; pred['current_flag_persistence'] = pred.current_bad
    for status in [-1, 0, 1]:
        g = pred if status == -1 else pred[pred.current_bad == status]
        w = balanced_weights(g.environment, g.episode); y = g.target_bad.to_numpy()
        for method in ['always_good','current_flag_persistence','dataset_rate','flag_transition',
                       'numeric_boosting_raw','multimodal_boosting_raw']:
            pp=g[method].to_numpy()
            quality_rows.append(dict(current_bad=status, method=method, frames=len(g),
                positives=int(y.sum()), environments=g.environment.nunique(),
                weighted_prevalence=float(np.dot(w,y)), brier=float(np.dot(w,(pp-y)**2)),
                average_precision=float(average_precision_score(y,pp,sample_weight=w)),
                auc=float(roc_auc_score(y,pp,sample_weight=w))))
    pd.DataFrame(quality_rows).to_csv(OUT/'quality_by_current_status.csv', index=False)
    audit = pd.read_csv(BASE/'record_audit.csv').set_index('episode'); videos=[]
    for path in sorted(Path('artifacts/c_video').glob('*__*.jsonl')):
        episode,view=path.stem.split('__'); meta=audit.loc[episode]
        detections=[len(json.loads(line)['detections']) for line in path.open()]
        videos.append(dict(dataset=meta.dataset, split=meta.split, environment=meta.environment,
            episode=episode, view=view, frames=len(detections),
            detected=int(np.count_nonzero(detections)), two_or_more=int(np.count_nonzero(np.array(detections)>=2))))
    vf=pd.DataFrame(videos); vf.to_csv(OUT/'visual_detection_by_video.csv', index=False)
    vs=vf.groupby(['dataset','split','view'])[['frames','detected','two_or_more']].sum().reset_index()
    vs['any_detection_fraction']=vs.detected/vs.frames;vs.to_csv(OUT/'visual_detection_summary.csv', index=False)
    files=[Path(__file__),BASE/'state_model.json',BASE/'multiscale_model.json',BASE/'quality_validation_predictions.csv']
    (OUT/'diagnostic_provenance.json').write_text(json.dumps(dict(
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope='train/validation only; exploratory review; v1 unchanged',
        scalar_coefficient=shrinkage),indent=2)+'\n')
    print('Scalar velocity coefficient:',shrinkage)
    print(summary[summary.context=='all'].to_string(index=False))
    print(coverage_summary[coverage_summary.nominal==.9].to_string(index=False))
    print(pd.DataFrame(quality_rows).to_string(index=False))


if __name__ == '__main__':main()
