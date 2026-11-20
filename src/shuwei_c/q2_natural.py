"""Descriptive natural-quality episodes; no position errors or interpolated truth.

Flags are observations at discrete timestamps. Event onset/recovery times are
bracketed by adjacent labels, and boundary episodes are explicitly censored.
AI-assisted: OpenAI Codex, 2026-11-20; verified through synthetic invariants.
"""
import numpy as np


def flag_runs(time, bad, recurrence_horizons=(.2, .5, 1.)):
    """Return event rows for a single hand's native annotation timestamps.

    ``detection_to_recovery_s`` starts at the first flagged timestamp, not an
    unobserved physical onset. The unobserved onset lies after the preceding
    good timestamp, recovery after the last flagged timestamp. Complete-event
    duration bounds describe that sampling uncertainty. Runs at either edge
    have no finite full-duration upper bound.
    """
    t, q = np.asarray(time, float), np.asarray(bad, bool)
    if t.ndim != 1 or q.shape != t.shape or len(t) < 2:
        raise ValueError('At least two matching time/flag samples are required')
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Quality timestamps must be finite and strictly increasing')
    starts = np.flatnonzero(q & ~np.r_[False, q[:-1]])
    ends = np.flatnonzero(q & ~np.r_[q[1:], False]) + 1
    rows = []
    for number, (start, stop) in enumerate(zip(starts, ends)):
        left_censored, right_censored = start == 0, stop == len(t)
        recovery = float(t[stop]) if not right_censored else np.nan
        followup_end = float(t[-1]) if right_censored else recovery
        lower = float(t[stop-1]-t[start])
        upper = (float(recovery-t[start-1])
                 if not left_censored and not right_censored else np.nan)
        next_start = int(starts[number+1]) if number+1 < len(starts) else None
        good_followup = ((float(t[next_start])-recovery if next_start is not None
                          else float(t[-1])-recovery) if not right_censored else np.nan)
        row = dict(event=number, start_index=int(start), stop_index=int(stop),
                   first_flagged_s=float(t[start]), last_flagged_s=float(t[stop-1]),
                   preceding_good_s=float(t[start-1]) if not left_censored else np.nan,
                   first_good_s=recovery, flagged_samples=int(stop-start),
                   left_censored=bool(left_censored), right_censored=bool(right_censored),
                   full_duration_lower_s=lower, full_duration_upper_s=upper,
                   detection_followup_s=followup_end-float(t[start]),
                   detection_to_recovery_s=recovery-float(t[start]) if not right_censored else np.nan,
                   recovery_observed=not right_censored,
                   next_flagged_s=float(t[next_start]) if next_start is not None else np.nan,
                   good_followup_s=good_followup,
                   good_spell_right_censored=not right_censored and next_start is None)
        for horizon in recurrence_horizons:
            key = f'recurrence_by_{horizon:g}s'
            if right_censored:
                row[key] = np.nan
            elif next_start is not None and good_followup <= horizon + 1e-9:
                row[key] = 1.
            elif good_followup >= horizon - 1e-9:
                row[key] = 0.
            else:
                # Follow-up ended before it could establish absence of recurrence.
                row[key] = np.nan
        rows.append(row)
    return rows


def record_statistics(time, bad, events):
    t, q = np.asarray(time, float), np.asarray(bad, bool)
    dt = np.diff(t)
    exposure = float(t[-1]-t[0])
    onsets = sum(not e['left_censored'] for e in events)
    # Accounting convention only: each observed flag labels its interval to
    # the next annotation. This does not impute positions or physical truth.
    flagged_interval_time = float(np.sum(dt*q[:-1]))
    good_interval_time = float(np.sum(dt*~q[:-1]))
    return dict(samples=len(t), flagged_samples=int(q.sum()),
                sample_flag_fraction=float(q.mean()), record_span_s=exposure,
                flagged_left_hold_s=flagged_interval_time,
                flagged_left_hold_fraction=flagged_interval_time/exposure,
                good_left_hold_s=good_interval_time,
                bad_run_count=len(events), observed_onsets=onsets,
                onsets_per_record_minute=60*onsets/exposure,
                onsets_per_good_minute=60*onsets/good_interval_time if good_interval_time else np.nan,
                observed_recoveries=sum(e['recovery_observed'] for e in events),
                left_censored_runs=sum(e['left_censored'] for e in events),
                right_censored_runs=sum(e['right_censored'] for e in events),
                records_with_bad=int(bool(events)),
                median_dt_s=float(np.median(dt)), max_dt_s=float(np.max(dt)))


def product_limit(durations, recovered):
    """Descriptive Kaplan-Meier curve for time to first observed recovery.

    Include only events with observed onset (exclude left-censored events).
    Right-censored runs remain at risk through their follow-up time. For ties,
    recoveries occur before censoring, hence censored observations at that time
    are included in the denominator. Independent censoring is an interpretation
    assumption, not established by this calculation; this is not physical-time
    survival or a causal estimate.
    """
    durations = np.asarray(durations, float)
    recovered = np.asarray(recovered, bool)
    if durations.shape != recovered.shape or durations.ndim != 1:
        raise ValueError('Duration and recovery arrays must match')
    if not np.isfinite(durations).all() or np.any(durations < 0):
        raise ValueError('Nonnegative finite follow-up required')
    rows, survival = [], 1.
    for t in np.unique(durations):
        risk = int(np.sum(durations >= t))
        at_time = durations == t
        events = int(np.sum(at_time & recovered))
        censored = int(np.sum(at_time & ~recovered))
        survival *= 1.-events/risk
        rows.append(dict(followup_s=float(t), at_risk=risk, recoveries=events,
                         censored=censored, probability_not_yet_recovered=float(survival)))
    return rows


def survival_at(curve, horizon):
    if not curve:
        return np.nan
    last = curve[-1]
    if horizon > last['followup_s'] and last['probability_not_yet_recovered'] > 0:
        return np.nan
    seen = [r for r in curve if r['followup_s'] <= horizon+1e-9]
    return seen[-1]['probability_not_yet_recovered'] if seen else 1.
