#!/usr/bin/env python3
"""Run local numerical development experiments; does not open the test trajectories.

AI: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset,DEFAULT_DATA,balanced_weights
from shuwei_c.motion import estimate_noise,calibration
from shuwei_c.quality import train_quality,evaluate_quality
from shuwei_c.q2 import evaluate_records
from shuwei_c.duration import fit_duration,DurationModel,save_model,service_samples
from shuwei_c.q3 import Replay,jobs_from_rows,sample_sources
from shuwei_c.policy import SchedulingPolicy


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run_q1q2(data,out,protocol):
    train=data.records('train');validation=data.records('validation')
    noise=estimate_noise(train)
    cal=calibration(train,noise,protocol['q2']['main_windows_s']+protocol['q2']['stress_windows_s'])
    write_json(out/'motion_model.json',dict(noise=noise,calibration=cal))
    model,params=train_quality(train);write_json(out/'quality_model.json',params)
    write_json(out/'q1_quality_validation.json',evaluate_quality(model,params,validation))
    # Train-side leave-operator-out stress test; official split remains untouched.
    operator_rows=[]
    for operator in sorted({r.operator for r in train}):
        held=[r for r in train if r.operator==operator];other=[r for r in train if r.operator!=operator]
        if not other:continue
        m,p=train_quality(other)
        operator_rows.extend([dict(held_operator=operator,**r) for r in evaluate_quality(m,p,held)])
    write_json(out/'q1_leave_operator_stress.json',operator_rows)
    rows=evaluate_records(validation,noise,cal,protocol['q2'])
    frame=pd.DataFrame(rows);frame.to_csv(out/'q2_validation_windows.csv',index=False)
    keys=['dataset','horizon_s','cap','delay_s','method','trigger','protocol']
    summary=[]
    for key,f in frame.groupby(keys):
        valid=f.error.notna();f=f[valid]
        if not len(f):continue
        weights=balanced_weights(f.environment,f.episode)
        metrics={c:float(np.average(f[c],weights=weights)) for c in
            ['error','coverage','width','interval_score','requests','responses_within_window','request_cost','latency_sum_s']}
        summary.append(dict(zip(keys,key),windows=len(f),**metrics))
    pd.DataFrame(summary).to_csv(out/'q2_validation_summary.csv',index=False)
    print(json.dumps(dict(stage='q1q2',windows=len(frame),summary_rows=len(summary))),flush=True)


def duration_validation(data,params):
    rows=[]
    for prefix in [False,True]:
        model=DurationModel(params,prefix)
        for unit in data.eligible('validation'):
            for level,h in enumerate([.2,.6,1.]):
                dist=model.distribution(unit['family_id'],data.unit_prefix(unit,h),level)
                samples=service_samples(dist,1.,0.,199)
                lo,hi=np.quantile(samples,[.05,.95]);truth=float(unit['duration_s'])
                rows.append(dict(episode=unit['episode_uuid'],environment=unit['environment_id'],
                    family=unit['family_id'],level=level,use_prefix=prefix,duration=truth,
                    median=float(np.median(samples)),abs_error=abs(float(np.median(samples))-truth),
                    covered=bool(lo<=truth<=hi),width=hi-lo,
                    interval_score=hi-lo+20*max(lo-truth,0)+20*max(truth-hi,0)))
    return rows


def run_q3(data,out,protocol,selected=None,seed_count=30):
    model_path=out/'duration_model.json';noquality_path=out/'duration_no_quality.json'
    if not model_path.exists():save_model(fit_duration(data),model_path)
    if not noquality_path.exists():save_model(fit_duration(data,use_quality=False),noquality_path)
    params=json.loads(model_path.read_text());no_quality=json.loads(noquality_path.read_text())
    pd.DataFrame(duration_validation(data,params)).to_csv(out/'q3_duration_validation.csv',index=False)
    configurations={
        'edd':dict(rule='edd',prefix=False),
        'weighted':dict(rule='weighted',prefix=False),
        'family_lookahead':dict(rule='lookahead',prefix=False),
        'prefix_no_review':dict(rule='lookahead',prefix=True),
        'prefix_voi':dict(rule='lookahead',prefix=True,review=True),
        'no_quality':dict(rule='lookahead',prefix=True,quality=False),
        'reserve_q50':dict(rule='lookahead',prefix=True,reservation='duration_q50'),
        'reserve_q80':dict(rule='lookahead',prefix=True,reservation='duration_q80')}
    if selected:configurations={k:v for k,v in configurations.items() if k in selected}
    for name,cfg in configurations.items():
        begin=time.monotonic();rows=[]
        with (out/f'q3_events_{name}.jsonl').open('w') as logs:
            for alpha,beta in protocol['q3']['mappings']:
                for case in data.conditions['cases']:
                    jobs=jobs_from_rows(data.jobs,case['case_id'])
                    for seed in data.conditions['validation_seeds'][:seed_count]:
                        sources=sample_sources(data,jobs,'validation',seed)
                        model=DurationModel(params if cfg.get('quality',True) else no_quality,cfg['prefix'])
                        policy=SchedulingPolicy(model,rule=cfg['rule'],review=cfg.get('review',False),reservation=cfg.get('reservation','minimum'))
                        replay=Replay(jobs,sources,data.unit_prefix,data.zones,case,data.conditions,alpha,beta)
                        metrics=replay.run(policy)
                        identity=dict(policy=name,case_id=case['case_id'],seed=seed,alpha=alpha,beta=beta,split='validation')
                        rows.append(dict(**identity,**metrics))
                        # Sources are written after termination, to local ignored audit output only.
                        logs.write(json.dumps(dict(**identity,events=replay.events,metrics=metrics,
                            source_audit={k:{f:u[f] for f in ['unit_id','duration_s','source_segment_index']} for k,u in replay.source_audit().items()}))+'\n')
        pd.DataFrame(rows).to_csv(out/f'q3_trials_{name}.csv',index=False)
        print(json.dumps(dict(stage='q3',policy=name,trials=len(rows),elapsed_s=round(time.monotonic()-begin,2),
                             invalid=sum(r['invalid_actions'] for r in rows))),flush=True)
    frames=[pd.read_csv(p) for p in sorted(out.glob('q3_trials_*.csv'))]
    if frames:
        frame=pd.concat(frames,ignore_index=True)
        numeric=['weighted_on_time_rate','weighted_completion_rate','weighted_tardiness_s','total_cost','observations',
                 'elapsed_s','observed_overrun_s','reserved_idle_s','censored_units','invalid_actions']
        summary=frame.groupby(['policy','case_id','alpha','beta'])[numeric].agg(['mean','std','count'])
        summary.columns=['_'.join(c) for c in summary.columns]
        summary.to_csv(out/'q3_validation_summary.csv')
        paired=[]
        baseline=frame[frame.policy=='edd']
        for name,f in frame.groupby('policy'):
            merged=f.merge(baseline,on=['case_id','alpha','beta','seed'],suffixes=('','_edd'))
            for key,g in merged.groupby(['case_id','alpha','beta']):
                d=(g.weighted_on_time_rate-g.weighted_on_time_rate_edd).to_numpy()
                rng=np.random.default_rng(20261120)
                means=np.mean(rng.choice(d,size=(2000,len(d)),replace=True),axis=1)
                paired.append(dict(policy=name,case_id=key[0],alpha=key[1],beta=key[2],n=len(d),
                    mean_difference=float(np.mean(d)),bootstrap_low=float(np.quantile(means,.025)),bootstrap_high=float(np.quantile(means,.975))))
        pd.DataFrame(paired).to_csv(out/'q3_paired_vs_edd.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output',type=Path,default=Path('artifacts/c_20261120'))
    parser.add_argument('--protocol',type=Path,default=Path('configs/c_protocol.json'))
    parser.add_argument('--stage',choices=['q1q2','q3','all'],default='all')
    parser.add_argument('--policies',nargs='+')
    parser.add_argument('--seed-count',type=int,default=30)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not 1<=args.seed_count<=30:parser.error('seed-count must be 1..30')
    protocol=json.loads(args.protocol.read_text());data=Dataset(args.data_root)
    manifest=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  stage=args.stage,seed_count=args.seed_count,protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                  input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.task_root.glob('*') if p.is_file()},
                  test_used=False,human_verification='pending')
    write_json(args.output/f'run_manifest_{args.stage}.json',manifest)
    if args.stage in ['q1q2','all']:run_q1q2(data,args.output,protocol)
    if args.stage in ['q3','all']:run_q3(data,args.output,protocol,args.policies,args.seed_count)


if __name__=='__main__':main()
