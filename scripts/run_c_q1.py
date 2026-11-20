#!/usr/bin/env python3
"""Question 1 development results with complete numerical and visual baselines.

AI-assisted: OpenAI Codex (GPT-6 family), 2026-11-20; release date unavailable.
Human verification pending. Raw references remain imperfect measured records.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import platform
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset,DEFAULT_DATA
from shuwei_c.motion import estimate_noise,filter_record
from shuwei_c.q1 import audit_records,state_evaluation,aggregate_state,fit_reliability,retrospective_segments


def write_json(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def bootstrap_parameters(records,output):
    # Re-estimate each record, then bootstrap environment means of log parameters.
    rows=[];rng=np.random.default_rng(20261120)
    for rec in records:
        params=estimate_noise([rec])[rec.dataset]
        rows.append(dict(dataset=rec.dataset,environment=rec.environment,episode=rec.episode,**params))
    frame=pd.DataFrame(rows);result=[]
    for ds,g in frame.groupby('dataset'):
        for p in ['r','acceleration','hand_scale']:
            env=np.log(g.set_index(['environment','episode'])[p]).groupby('environment').mean().to_numpy()
            samples=np.exp(np.mean(rng.choice(env,size=(4000,len(env))),axis=1))
            result.append(dict(dataset=ds,parameter=p,estimate=float(np.exp(env.mean())),
                low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975)),environments=len(env)))
    frame.to_csv(output/'noise_record_estimates.csv',index=False)
    pd.DataFrame(result).to_csv(output/'noise_parameter_bootstrap.csv',index=False)


def state_models(train,validation,out):
    noise=estimate_noise(train);grid=[];models=[]
    for qscale,rscale,bad_multiplier in itertools.product([.1,1.,10.],[1.,10.],[1.,10.,100.]):
        candidate={ds:dict(v,acceleration=v['acceleration']*qscale,r=v['r']*rscale,bad_multiplier=bad_multiplier) for ds,v in noise.items()}
        frame=state_evaluation(validation,candidate,methods=['filter'],horizons=[.1])
        frame=frame[frame.context=='all'].copy()
        frame['normalized_error']=[v/noise[ds]['hand_scale'] for ds,v in zip(frame.dataset,frame.mean_euclidean_error)]
        per_episode=frame.groupby(['environment','episode']).normalized_error.mean()
        score=float(per_episode.groupby('environment').mean().mean())
        models.append(candidate);grid.append(dict(q_scale=qscale,r_scale=rscale,bad_multiplier=bad_multiplier,
            validation_normalized_error=score))
    best=int(np.argmin([r['validation_normalized_error'] for r in grid]));selected=models[best]
    write_json(out/'state_model.json',dict(base_training_noise=noise,selected_noise=selected,
        selection=grid[best],criterion='environment/episode-balanced 0.1s forecast Euclidean error / training hand scale',
        status='validation-selected development model; test untouched'))
    pd.DataFrame(grid).to_csv(out/'state_parameter_sensitivity.csv',index=False)
    baseline=state_evaluation(validation,selected)
    default=state_evaluation(validation,noise,methods=['filter']);default['method']='filter_default'
    result=pd.concat([baseline,default],ignore_index=True)
    result.to_csv(out/'state_validation_by_record.csv',index=False)
    aggregate_state(result).to_csv(out/'state_validation_summary.csv',index=False)
    # Export the same centroids, velocities and derived relative vector for Q2.
    states_dir=out/'states';states_dir.mkdir(exist_ok=True)
    for rec in [*train,*validation]:
        t,p,bad=rec.hands();state,cov=filter_record(rec,selected[rec.dataset])
        np.savez_compressed(states_dir/f'{rec.episode}.npz',time=t,centroid=state[:,:,0],velocity=state[:,:,1],
            relative_right_minus_left=state[:,1,0]-state[:,0,0],covariance=cov,raw_bad=bad,
            observed_centroid=p.mean(axis=2))
    bootstrap_parameters(train,out)
    print(json.dumps(dict(q1_state_selection=grid[best])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    p.add_argument('--output',type=Path,default=Path('artifacts/c_q1_v1'))
    p.add_argument('--video-root',type=Path,default=Path('artifacts/c_video'))
    p.add_argument('--stage',choices=['state','quality','all'],default='all')
    p.add_argument('--without-video',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=Dataset(args.data_root);train=data.records('train');validation=data.records('validation')
    if args.stage in ['state','all']:
        audit,intervals,duplicates=audit_records([*train,*validation])
        audit.to_csv(args.output/'record_audit.csv',index=False)
        intervals.to_csv(args.output/'interval_uses.csv',index=False)
        write_json(args.output/'duplicate_trajectory_groups.json',duplicates)
        retrospective_segments(data,[*train,*validation]).to_csv(args.output/'retrospective_stage_statistics.csv',index=False)
        state_models(train,validation,args.output)
    if args.stage in ['quality','all']:
        if not args.without_video and not (args.video_root/'video_manifest.json').exists():
            raise RuntimeError('Video extraction is incomplete. Complete it or explicitly request --without-video.')
        fit_reliability(train,validation,args.output,None if args.without_video else args.video_root)
    write_json(args.output/f'run_manifest_{args.stage}.json',dict(date='2026-11-20',python=platform.python_version(),
        numpy=np.__version__,pandas=pd.__version__,test_used=False,train_records=len(train),validation_records=len(validation),
        input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.task_root.iterdir() if p.is_file()},
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path('src/shuwei_c').glob('*.py'),Path(__file__)]}))


if __name__=='__main__':main()
