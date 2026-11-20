#!/usr/bin/env python3
"""Q2 natural-quality summaries and independent development-result audit.

Only train/validation quality channels are analysed. This script deliberately
does not report positional error within natural low-quality intervals.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset
from shuwei_c.q2_natural import flag_runs, record_statistics, product_limit, survival_at

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/c_q2/natural'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hierarchical_mean(frame, column):
    """Independent explicit loops: equal environments, records, and hand rows."""
    environment_values = []
    for environment in sorted(frame.environment.unique()):
        ef = frame[frame.environment == environment]
        episode_values = []
        for episode in sorted(ef.episode.unique()):
            values = ef.loc[ef.episode == episode, column].to_numpy(float)
            finite = values[np.isfinite(values)]
            if len(finite):
                episode_values.append(float(np.mean(finite)))
        if episode_values:
            environment_values.append(float(np.mean(episode_values)))
    return float(np.mean(environment_values)) if environment_values else np.nan


def audit_windows(data):
    path = ROOT/'artifacts/c_q2/protocol/development_windows.csv'
    windows = pd.read_csv(path)
    failures = []
    for row in windows.itertuples(index=False):
        rec = data.record(row.episode)
        t, p, bad = rec.hands()
        start, stop = int(row.origin_index)+1, int(row.stop)
        checks = {
            'allowed_split': rec.split in ('train', 'validation'),
            'source_metadata': rec.dataset == row.dataset and rec.environment == row.environment,
            'nonempty': stop > start,
            'qualified': not bad[start:stop].any() and np.isfinite(p[start:stop]).all(),
            'origin_timestamp': abs(t[start-1]-row.origin_s) < 1e-9,
            'targets_inside': np.all(t[start:stop] > row.origin_s) and np.all(t[start:stop] <= row.origin_s+row.horizon_s+1e-9),
            'complete_end': row.origin_s+row.horizon_s <= t[-1]+1e-9,
            'sample_count': stop-start == row.target_times,
        }
        for check, valid in checks.items():
            if not valid:
                failures.append(dict(window_id=row.window_id, check=check))
    return dict(rows=len(windows), duplicate_keys=int(windows.window_id.duplicated().sum()),
                splits=sorted(windows.split.unique().tolist()), failures=failures,
                source_sha256=sha(path))


def audit_baseline_summary():
    folder = ROOT/'artifacts/c_q2/baselines'
    manifest = folder/'validation_manifest.json'
    if not manifest.exists():
        return dict(status='not_run', reason='complete validation baseline manifest absent')
    frame = pd.read_csv(folder/'validation_windows.csv')
    summary = pd.read_csv(folder/'validation_summary.csv')
    keys = ['split', 'horizon_s', 'cap', 'delay_s', 'phase', 'method', 'policy', 'protocol']
    metrics = ['mean_error', 'normalized_error', 'unobserved_error', 'requests',
               'total_cost', 'waiting_union_s', 'latency_sum_s', 'current_direct_fraction']
    max_abs = {m: 0. for m in metrics}
    failures = []
    grouped = {k: g for k, g in frame.groupby(keys, sort=False, observed=True)}
    for row in summary.to_dict('records'):
        group = grouped[tuple(row[k] for k in keys)]
        if row['dataset'] != 'ALL':
            group = group[group.dataset == row['dataset']]
        for metric in metrics:
            value = hierarchical_mean(group, metric)
            difference = abs(value-row[metric])
            if np.isnan(value) and np.isnan(row[metric]):
                difference = 0.
            max_abs[metric] = max(max_abs[metric], float(difference))
            if not difference <= 1e-10:
                failures.append(dict(metric=metric, condition={k: row[k] for k in keys}, dataset=row['dataset']))
    violations = {
        'budget': int((frame.requests > frame.cap).sum()),
        'cost': int((np.abs(frame.total_cost-frame.requests) > 1e-10).sum()),
        'latency_sum': int((np.abs(frame.latency_sum_s-frame.requests*frame.delay_s) > 1e-10).sum()),
        'wait_negative': int((frame.waiting_union_s < -1e-10).sum()),
        'wait_exceeds_window': int((frame.waiting_union_s > frame.horizon_s+1e-9).sum()),
        'wait_exceeds_latency_sum': int((frame.waiting_union_s > frame.latency_sum_s+1e-9).sum()),
        'unexpected_late_response': int((frame.late_responses > 0).sum()),
    }
    return dict(status='complete', windows_rows=len(frame), summary_rows=len(summary),
                duplicate_keys=int(frame.duplicated(['window_id']+keys).sum()),
                max_absolute_metric_recalculation_difference=max_abs, failures=failures,
                constraint_violations=violations,
                inputs={str(p.relative_to(ROOT)): sha(p) for p in (manifest, folder/'validation_windows.csv', folder/'validation_summary.csv')})


def audit_kinematic_samples(data):
    """Recompute representative curves without q2 replay or predictor imports."""
    folder = ROOT/'artifacts/c_q2/baselines'
    if not (folder/'validation_manifest.json').exists():
        return dict(status='not_run')
    frame = pd.read_csv(folder/'validation_windows.csv')
    windows = pd.read_csv(ROOT/'artifacts/c_q2/protocol/development_windows.csv').set_index('window_id')
    selected = frame[(frame.horizon_s == 1.) & (frame.cap == 2) & (frame.delay_s == .1)
                     & frame.method.isin(['constant', 'linear', 'damped_0.2'])]
    selected = selected.groupby(['episode', 'method'], sort=False).head(1)
    maximum = 0.
    checks = []
    for row in selected.itertuples(index=False):
        win = windows.loc[row.window_id]
        rec = data.record(row.episode)
        t, p, bad = rec.hands()
        centres = p.mean(axis=2)
        start, stop = int(win.origin_index)+1, int(win.stop)
        useful = t[stop-1]-.1-win.origin_s
        targets = [win.origin_s+(k+row.phase)*useful/2 for k in range(2)]
        requested = sorted(set(max(start, int(np.searchsorted(t, target-1e-9))) for target in targets))
        requested = [i for i in requested if i < stop and t[i]+.1 <= t[stop-1]+1e-9]
        estimates = []
        for i in range(start, stop):
            disclosed = list(range(start))+[j for j in requested if t[j]+.1 <= t[i]+1e-9]
            hands = []
            for hand in range(2):
                known = [j for j in disclosed if not bad[j, hand] and np.isfinite(centres[j, hand]).all()]
                if not known:
                    known = [j for j in disclosed if np.isfinite(centres[j, hand]).all()]
                last = known[-1]
                previous = known[-2] if len(known) > 1 else last
                v = (centres[last, hand]-centres[previous, hand])/max(t[last]-t[previous], 1e-9)
                age = t[i]-t[last]
                displacement_time = 0. if row.method == 'constant' else age
                if row.method.startswith('damped_'):
                    tau = float(row.method.split('_')[1])
                    displacement_time = tau*(1.-np.exp(-age/tau))
                hands.append(centres[last, hand]+v*displacement_time)
            estimates.append(hands)
        error = float(np.linalg.norm(np.array(estimates)-centres[start:stop], axis=-1).mean())
        difference = abs(error-row.mean_error)
        maximum = max(maximum, difference)
        checks.append(dict(window_id=row.window_id, dataset=row.dataset, method=row.method,
                           recorded_error=row.mean_error, independently_computed_error=error,
                           absolute_difference=difference))
    pd.DataFrame(checks).to_csv(OUT/'kinematic_independent_checks.csv', index=False)
    return dict(status='complete', curves=len(checks), max_absolute_difference=maximum,
                all_passed=maximum < 1e-10)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    data = Dataset()
    records = data.records('train')+data.records('validation')
    events, record_rows, quality_sources = [], [], []
    for rec in records:
        quality = rec.streams['quality']
        time_values, bad = quality['time'], quality['bad']
        quality_sources.append(dict(episode=rec.episode, dataset=rec.dataset, split=rec.split,
                                    time_sha256=hashlib.sha256(np.asarray(time_values, '<f8').tobytes()).hexdigest(),
                                    flags_sha256=hashlib.sha256(np.asarray(bad, np.uint8).tobytes()).hexdigest(),
                                    samples=len(time_values)))
        for hand, side in enumerate(('left', 'right')):
            rows = flag_runs(time_values, bad[:, hand])
            meta = dict(episode=rec.episode, environment=rec.environment,
                        dataset=rec.dataset, split=rec.split, hand=side)
            events.extend([dict(**meta, **row) for row in rows])
            record_rows.append(dict(**meta, **record_statistics(time_values, bad[:, hand], rows)))
    events = pd.DataFrame(events)
    record_rows = pd.DataFrame(record_rows)
    events.to_csv(OUT/'natural_events.csv', index=False)
    record_rows.to_csv(OUT/'record_hand_statistics.csv', index=False)
    pd.DataFrame(quality_sources).to_csv(OUT/'quality_source_fingerprints.csv', index=False)
    # Rates retain zero-event records, both hands and the environment hierarchy.
    rate_columns = ['sample_flag_fraction', 'flagged_left_hold_fraction',
                    'onsets_per_record_minute', 'onsets_per_good_minute', 'records_with_bad']
    summaries, event_summaries, curves, checkpoints = [], [], [], []
    for split in ('train', 'validation'):
        for dataset in ['ALL', 'EgoDemo', 'EgoPro', 'EgoStandard']:
            for hand in ['both', 'left', 'right']:
                rf = record_rows[record_rows.split == split]
                ef = events[events.split == split]
                if dataset != 'ALL':
                    rf, ef = rf[rf.dataset == dataset], ef[ef.dataset == dataset]
                if hand != 'both':
                    rf, ef = rf[rf.hand == hand], ef[ef.hand == hand]
                meta = dict(split=split, dataset=dataset, hand=hand)
                row = dict(**meta, records=rf.episode.nunique(), environments=rf.environment.nunique(),
                           hand_records=len(rf), samples=int(rf.samples.sum()), flagged_samples=int(rf.flagged_samples.sum()),
                           runs=int(rf.bad_run_count.sum()), observed_onsets=int(rf.observed_onsets.sum()),
                           observed_recoveries=int(rf.observed_recoveries.sum()),
                           left_censored_runs=int(rf.left_censored_runs.sum()), right_censored_runs=int(rf.right_censored_runs.sum()))
                row.update({column: hierarchical_mean(rf, column) for column in rate_columns})
                summaries.append(row)
                complete = ef[~ef.left_censored & ~ef.right_censored]
                erow = dict(**meta, runs=len(ef), complete_runs=len(complete),
                            event_environment_count=ef.environment.nunique(), event_record_count=ef.episode.nunique())
                for name, values in [('observed_span', ef.full_duration_lower_s),
                                     ('complete_detection_to_recovery', complete.detection_to_recovery_s)]:
                    erow[name+'_median_s'] = float(values.median()) if len(values) else np.nan
                    erow[name+'_p90_s'] = float(values.quantile(.9)) if len(values) else np.nan
                    erow[name+'_max_s'] = float(values.max()) if len(values) else np.nan
                for horizon in (.2, .5, 1.):
                    values = ef[f'recurrence_by_{horizon:g}s'].dropna()
                    erow[f'recurrence_eligible_{horizon:g}s'] = len(values)
                    erow[f'recurrence_fraction_{horizon:g}s'] = float(values.mean()) if len(values) else np.nan
                event_summaries.append(erow)
                onset = ef[~ef.left_censored]
                curve = product_limit(onset.detection_followup_s, onset.recovery_observed)
                curves.extend(dict(**meta, **point) for point in curve)
                for horizon in (0., .05, .1, .2, .5, 1., 2.):
                    checkpoints.append(dict(**meta, horizon_s=horizon, onset_count=len(onset),
                                            recovered_count=int(onset.recovery_observed.sum()),
                                            not_yet_recovered=survival_at(curve, horizon)))
    pd.DataFrame(summaries).to_csv(OUT/'quality_summary.csv', index=False)
    pd.DataFrame(event_summaries).to_csv(OUT/'event_duration_summary.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT/'recovery_product_limit.csv', index=False)
    pd.DataFrame(checkpoints).to_csv(OUT/'recovery_checkpoints.csv', index=False)
    audit = dict(window_selection=audit_windows(data), baseline_summary=audit_baseline_summary(),
                 kinematic_recalculation=audit_kinematic_samples(data))
    (OUT/'independent_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    definitions = dict(
        population='Official train and validation records only; native supplied quality annotations; no positional error for natural low-quality intervals.',
        bad_flag='Supplied per-hand bad flags decoded by data.py; unassigned bad flags in EgoPro/EgoStandard conservatively apply to both hands.',
        duration='Sampling brackets for a run of consecutive bad annotations: lower = last flagged minus first flagged timestamp; complete upper = first subsequent good minus preceding good timestamp. Interpreting these as continuous-duration bounds assumes no unobserved reversal inside an adjacent-sample interval. They are not independently verified physical-occlusion bounds. At either record boundary the full upper bound is unavailable.',
        observed_recovery='First good annotation after the flagged run; does not certify physical accuracy or permanent recovery.',
        discrete_latency='Detection-to-recovery is first subsequent good minus first flagged timestamp. It is distinct from unobserved physical onset-to-recovery duration.',
        occurrence='Onsets require a preceding observed good label; left-boundary runs do not count as observed onsets.',
        occupancy='Both sample fraction and duration-weighted left-held flag occupancy are reported. The latter is a flag accounting convention, not position interpolation.',
        rate_weighting='Equal hands per record, equal records per environment, equal environments, including records with zero bad flags.',
        event_weighting='Duration quantiles, recurrence proportions and product-limit curves are descriptive event-pooled quantities; environment/record/event counts are reported. No independent-event confidence interval is asserted.',
        survival='Product-limit time from first flagged annotation to first good annotation; left-boundary runs excluded; right-boundary runs censored. Unobserved tail is NaN unless all events recovered. Independent censoring is an assumption, not empirically established.',
        recurrence='Next first-bad label after recovery; a negative recurrence outcome requires at least the full horizon of follow-up. Short terminal follow-up is missing, not stable recovery.',
        reference='Kaplan EL, Meier P. Nonparametric Estimation from Incomplete Observations. JASA 53(282):457-481, 1958. https://doi.org/10.1080/01621459.1958.10501452')
    (OUT/'definitions.json').write_text(json.dumps(definitions, indent=2)+'\n')
    source_paths = [Path(__file__), ROOT/'src/shuwei_c/q2_natural.py', ROOT/'src/shuwei_c/data.py']
    manifest = dict(test_used=False, elapsed_s=time.perf_counter()-started, records=len(records),
                    environments=len(set(r.environment for r in records)), hand_records=len(record_rows),
                    natural_runs=len(events), no_low_quality_position_error=True,
                    source_hashes={str(p.relative_to(ROOT)): sha(p) for p in source_paths},
                    output_hashes={p.name: sha(p) for p in sorted(OUT.glob('*.csv'))})
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
