#!/usr/bin/env python3
"""Read-only Q1 review audit; new diagnostics never refit or select frozen models.

All splits: identity, completeness, continuity and retention accounting only.
Validation: matched-window constant baseline, descriptive errors/failure examples.
Existing test performance files are neither recalculated nor used for selection.
"""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset, csv_rows
from shuwei_c.q1 import audit_records, contiguous_runs
from shuwei_c.q1v2_state import matched_targets
from shuwei_c.q1v3_state import score

OUT = Path('artifacts/c_q1_review_repair')
FROZEN = Path('artifacts/c_q1_v3/frozen_numeric')
SOURCES = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    p = Path(p); SOURCES[str(p)] = sha(p)
    return pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)


def runs(mask, time):
    gaps = np.diff(time) > 1.5 * np.median(np.diff(time))
    return sum(1 + int(gaps[a:b-1].sum()) for a, b in contiguous_runs(mask))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((FROZEN/'freeze.json').read_text())
    protected = {**freeze['source_hashes'], **freeze['inherited_v2_hashes']}
    assert all(sha(p) == h for p, h in protected.items())
    data = Dataset(); records = [data.record(r['episode_uuid']) for r in data.split_rows]
    audit, intervals, duplicates = audit_records(records)
    audit.to_csv(OUT/'record_audit.csv', index=False)
    # The inherited interval 'use' column describes an earlier filter. Re-label
    # it explicitly for the selected point/quality models; never copy that claim.
    intervals = intervals.drop(columns='use')
    intervals['use'] = np.where(intervals.error_reference_allowed,
        'qualified_reference; training_fit_if_training_split', 'quality_analysis; no_position_error_reference')
    intervals.to_csv(OUT/'hand_quality_intervals.csv', index=False)
    meta = pd.DataFrame(data.split_rows); identity = []; flow = []; timing = []; purposes = []
    for rec in records:
        if rec.dataset == 'EgoDemo':
            m = pd.read_parquet(rec.folder/'meta/episodes/chunk-000/file-000.parquet').iloc[0]
            scene, uid = m.scene_id, m.episode_uuid
            assert m.environment_id == rec.environment and m.operator_id == rec.operator
        else:
            m = {r['field']: r['value'] for r in csv_rows(rec.folder/'session_metadata.csv')}
            scene, uid = m['task_info.scene_id'], m['task_info.episode_uuid']
            assert m['task_info.environment_id'] == rec.environment and m['operator.operator_id'] == rec.operator
        assert uid == rec.episode
        identity.append(dict(episode=uid, dataset=rec.dataset, split=rec.split,
                             environment=rec.environment, scene=scene, operator=rec.operator))
        t, p, bad = rec.hands(); dt = np.diff(t)
        for value in dt:
            timing.append(dict(dataset=rec.dataset, split=rec.split, episode=uid, dt_s=value))
        i, j = matched_targets(t, .1)
        warm = (t >= t[0]+.4-1e-9) & (np.arange(len(t)) >= 2)
        boundary = int((t[warm]+.1 > t[-1]+1e-9).sum())
        excluded = np.ones((len(t),2), bool); excluded[i] = False
        qualified = np.zeros_like(excluded); quality = np.zeros_like(excluded)
        qualified[i] = ~bad[j]; quality[i] = bad[j]
        f = dict(episode=uid, dataset=rec.dataset, split=rec.split,
                 timestamps=len(t), hand_observations=2*len(t), warmup=2*int((~warm).sum()),
                 boundary=2*boundary, mismatch=2*int(warm.sum()-boundary-len(i)),
                 qualified=int(qualified.sum()), flagged_targets=int(quality.sum()),
                 flagged_observations=int(bad.sum()))
        assert f['hand_observations'] == f['warmup']+f['boundary']+f['mismatch']+f['qualified']+f['flagged_targets']
        flow.append(f)
        for name, mask in [('qualified_targets', qualified), ('flagged_targets', quality),
                           ('excluded_prediction_origins', excluded), ('flagged_observations', bad)]:
            purposes.append(dict(episode=uid, dataset=rec.dataset, split=rec.split, purpose=name,
                                 hand_samples=int(mask.sum()), hand_intervals=sum(runs(mask[:,h],t) for h in range(2))))
    identity = pd.DataFrame(identity); identity.to_csv(OUT/'identity_audit.csv', index=False)
    overlaps = []
    for field in ['episode','environment','scene','operator']:
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            x=set(identity.loc[identity.split.eq(a),field]); y=set(identity.loc[identity.split.eq(b),field])
            overlaps.append(dict(field=field, split_a=a, split_b=b, common=len(x & y), a_unique=len(x), b_unique=len(y)))
    pd.DataFrame(overlaps).to_csv(OUT/'split_overlap.csv', index=False)
    pd.DataFrame(flow).to_csv(OUT/'retention_by_record.csv', index=False)
    retention=pd.DataFrame(flow).groupby('split').sum(numeric_only=True)
    retention.to_csv(OUT/'retention_by_split.csv')
    # Check retained test identity keys only. Never read its numerical errors or
    # predictions here, and do not run an inference/evaluation on test records.
    frozen_test_path=FROZEN/'test/points_h0.100000.parquet'
    SOURCES[str(frozen_test_path)]=sha(frozen_test_path)
    frozen_test=pd.read_parquet(frozen_test_path,columns=['episode','origin','target','hand'])
    expected=[]
    for rec in data.records('test'):
        t,p,bad=rec.hands();i,j=matched_targets(t,.1)
        for a,b in zip(i,j):
            for h in range(2):
                if not bad[b,h]:expected.append(dict(episode=rec.episode,origin=int(a),target=int(b),hand=h))
    cols=['episode','origin','target','hand']
    pd.testing.assert_frame_equal(pd.DataFrame(expected)[cols].sort_values(cols).reset_index(drop=True),
        frozen_test[cols].sort_values(cols).reset_index(drop=True),check_dtype=False)
    purposes=pd.DataFrame(purposes)
    summary=purposes.groupby(['split','purpose']).agg(hand_samples=('hand_samples','sum'),
        hand_intervals=('hand_intervals','sum'),records=('hand_samples',lambda x:int((x>0).sum()))).reset_index()
    summary.to_csv(OUT/'purpose_counts.csv',index=False)
    timing=pd.DataFrame(timing)
    timing.groupby(['dataset','split']).agg(intervals=('dt_s','size'),dt_min=('dt_s','min'),
        dt_p01=('dt_s',lambda x:x.quantile(.01)),dt_median=('dt_s','median'),
        dt_p99=('dt_s',lambda x:x.quantile(.99)),dt_max=('dt_s','max')).to_csv(OUT/'sampling_intervals.csv')
    pooled=timing.groupby('dataset').agg(intervals=('dt_s','size'),p01_ms=('dt_s',lambda x:1000*x.quantile(.01)),
        median_ms=('dt_s',lambda x:1000*x.median()),p99_ms=('dt_s',lambda x:1000*x.quantile(.99)))
    pooled['gap_count']=audit.groupby('dataset').gaps_over_1_5dt.sum()
    pooled.to_csv(OUT/'sampling_distribution.csv')
    # Constant predictor evaluated only at the exact frozen validation targets.
    metrics=[]; all_rows=[]
    for source in sorted((FROZEN/'validation').glob('points_h*.parquet')):
        f=read(source); f['constant_error']=np.nan
        for episode, ids in f.groupby('episode').groups.items():
            rec=data.record(episode); t,p,bad=rec.hands(); c=p.mean(axis=2)
            g=f.loc[ids]; ii=g.origin.to_numpy(int); jj=g.target.to_numpy(int); hh=g.hand.to_numpy(int)
            assert (~bad[jj,hh]).all()
            f.loc[ids,'constant_error']=np.linalg.norm(c[jj,hh]-c[ii,hh],axis=1)
        for ds in ['ALL','EgoDemo','EgoPro','EgoStandard']:
            g=f if ds=='ALL' else f[f.dataset.eq(ds)]
            for method in ['constant','linear','scalar','multiscale','gate']:
                metrics.append(dict(horizon=float(f.horizon.iloc[0]),dataset=ds,method=method,
                    mean_error=score(g,g[method+'_error']),n=len(g),environments=g.environment.nunique()))
        f[['episode','environment','dataset','split','origin','target','origin_s','target_s','hand','horizon',
           'constant_error','linear_error','multiscale_error','gate_error']].to_parquet(OUT/source.name,index=False)
        all_rows.append(f)
    metrics=pd.DataFrame(metrics); metrics.to_csv(OUT/'validation_point_metrics.csv',index=False)
    old=read(FROZEN/'validation/point_metrics.csv')
    check=metrics.merge(old,on=['horizon','dataset','method'],suffixes=('_audit','_frozen'))
    assert np.max(abs(check.mean_error_audit-check.mean_error_frozen)) < 1e-12
    f=next(f for f in all_rows if f.horizon.iloc[0]==.1)
    env=f.groupby(['environment','episode','hand'])[['constant_error','linear_error','multiscale_error','gate_error']].mean()
    env=env.groupby(['environment','episode']).mean().groupby('environment').mean().reset_index()
    dev=audit[~audit.split.eq('test')].copy()
    dev['flagged_fraction']=(dev.left_bad_frames+dev.right_bad_frames)/dev.total_hand_frames
    flags=dev.groupby(['split','environment']).flagged_fraction.mean().reset_index()
    env=env.merge(flags[flags.split.eq('validation')],on='environment')
    env.to_csv(OUT/'validation_environment_descriptives.csv',index=False)
    sem=read('artifacts/c_q1_v2/semantic_segments.csv'); desc=[]
    for action in ['reach','pick','place','transport','arrange','open_close','unknown']:
        g=sem[sem[action]>0]
        # Descriptive weighted means of the existing phase summaries. Overlapping
        # labels are intentionally multi-membership; not exclusive event counts.
        desc.append(dict(action=action,phase_intervals=len(g),episodes=g.episode.nunique(),
            environments=g.environment.nunique(),timestamps=int(g.frames.sum()),
            flagged_percent=100*np.average(g.bad_fraction,weights=g.frames) if len(g) else np.nan,
            mean_log1p_speed=np.average(g.mean_log_speed,weights=g.qualified_frames) if len(g) and g.qualified_frames.sum() else np.nan))
    pd.DataFrame(desc).to_csv(OUT/'semantic_descriptives.csv',index=False)
    # Existing training-OOF calibrated baseline radii across five horizons. The
    # selected conditional-feedback model is calibrated at 0.1 s only.
    radii=read('artifacts/c_q1_v2/state_set_validation.csv')
    radii.query("method == 'multiscale' and context == 'all' and scheme == 'adaptive' and nominal == 0.9").to_csv(OUT/'horizon_baseline_radii.csv',index=False)
    from shuwei_c.q1v2_quality import event_metrics
    onset=read('artifacts/c_q1_v3/onset_h0.5/validation.parquet')
    event_table=read('artifacts/c_q1_v3/onset_h0.5/events.csv')
    event_rows=[]; event_summary=[]; envs=sorted(onset.environment.unique())
    draws=np.random.default_rng(20261120).integers(0,len(envs),(10000,len(envs)))
    for method in ['numeric_leaves7','diagnostics_leaves7']:
        threshold=float(event_table.loc[event_table.method.eq(method)&event_table.nominal_false_alarm.eq(.05),'threshold'].iloc[0])
        p=onset[method+'_onset_probability'].to_numpy()
        score_event, events=event_metrics(data.records('validation'),onset,np.c_[1-p,p,np.zeros((len(p),2))],threshold,.5)
        events['method']=method;event_rows.append(events)
        counts=events.groupby('environment').detected.agg(['sum','size']).reindex(envs,fill_value=0)
        num=counts['sum'].to_numpy()[draws].sum(1); den=counts['size'].to_numpy()[draws].sum(1)
        rates=num[den>0]/den[den>0]
        event_summary.append(dict(method=method,**score_event,recall_low=np.quantile(rates,.025),
            recall_high=np.quantile(rates,.975),bootstrap_environments=len(envs),undefined_resamples=int((den==0).sum())))
    pd.concat(event_rows).to_csv(OUT/'validation_onset_event_details.csv',index=False)
    pd.DataFrame(event_summary).to_csv(OUT/'validation_onset_event_uncertainty.csv',index=False)
    facts=dict(records=len(records),uuid_unique=identity.episode.nunique(),uuid_duplicates=int(identity.episode.duplicated().sum()),
        trajectory_duplicate_groups=len(duplicates),nonfinite_hand_samples=int(audit.nonfinite_hand_frames.sum()),
        gaps_over_1_5dt=int(audit.gaps_over_1_5dt.sum()),nonincreasing_time=0,
        quality_exact=bool(audit.quality_time_exact.all()),head_exact=bool(audit.head_time_exact.all()),
        frozen_validation_metric_max_difference=float(np.max(abs(check.mean_error_audit-check.mean_error_frozen))),
        test_retention_keys_match_frozen=True,test_performance_recomputed=False,models_fitted=False,coordinate_unit='unverified native coordinates',
        unit_evidence='operator.operator_info.unit=m describes anthropometry only; no coordinate scale/axis documentation is present',
        hand_scales=freeze['noise'],source_sha256=SOURCES,protected_source_sha256=protected)
    assert all(sha(p)==h for p,h in protected.items())
    (OUT/'audit.json').write_text(json.dumps(facts,indent=2)+'\n')
    print(json.dumps({k:v for k,v in facts.items() if k not in ['source_sha256','protected_source_sha256','hand_scales']},indent=2))
    print(retention.to_string()); print(summary.to_string(index=False))


if __name__=='__main__':main()
