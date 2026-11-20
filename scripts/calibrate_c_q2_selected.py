#!/usr/bin/env python3
"""Validation-only development of radii for the fixed equal-weight gate blend.

Radius cross-fitting below is a training device for an empirical correction.
It is NOT reported as an unbiased OOF score of the whole point/radius pipeline:
the upstream point OOF models are shared. Official validation is excluded from
every point fit, radius fit and correction. Whole-pipeline outer auditing is a
separate required experiment before final claims.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q2_learning import load_pack
from shuwei_c.q2_features import transform_features
from shuwei_c.q2_calibration import RadiusCalibrator,AgeBinnedRadius,evaluate_paths_long
from shuwei_c.q2_radius_refine import AgeRatioCorrection


def point_errors(pack,split):
    errors=np.full(len(pack['meta']),np.nan);paths=[]
    keys=['episode','window_id','scenario','time_s','hand']
    for number in (range(4) if split=='train' else [-1]):
        folder='full' if number==-1 else f'fold{number}'
        a=Path(f'artifacts/c_q2/static_gate/nested/{folder}/ensemble_predictions.parquet')
        b=Path(f'artifacts/c_q2/gate_oof_main/residual/{folder}/predictions.parquet')
        paths.extend([a,b]);f=pd.read_parquet(a);g=pd.read_parquet(b)
        take=pack['meta'].environment.isin(f.environment.unique()).to_numpy()
        if not f[keys].equals(g[keys]) or not f[keys].equals(pack['meta'].loc[take,keys].reset_index(drop=True)):
            raise ValueError('Point-model row identity differs')
        prediction=(f[['prediction_'+c for c in 'xyz']].to_numpy(float)+g[['prediction_'+c for c in 'xyz']].to_numpy(float))/2
        reference=pack['anchor'][take].astype(float)+pack['target'][take].astype(float)
        errors[take]=np.linalg.norm(prediction-reference,axis=1)
    if not np.isfinite(errors).all():raise ValueError('Incomplete point residuals')
    return errors,paths


def sample(meta,environments,per_group):
    allowed=meta.environment.isin(environments).to_numpy()
    rank=np.random.default_rng(20261120).random(len(meta));chosen=[]
    for _,group in meta[allowed].groupby(['episode','horizon_s','scenario','hand'],sort=True):
        indices=group.index.to_numpy();chosen.extend(indices[np.argsort(rank[indices])[:per_group]])
    return np.array(sorted(chosen),int)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('artifacts/c_q2/calibration/hybrid_development'))
    parser.add_argument('--levels',nargs='+',type=float,default=[.5,.8,.9,.95])
    parser.add_argument('--fit-per-group',type=int,default=64)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('radius_evaluation','scripts/analyze_c_q2_calibration.py')
    evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
    train=load_pack('artifacts/c_q2/learning_data/train');val=load_pack('artifacts/c_q2/learning_data/validation')
    train_error,files=point_errors(train,'train');val_error,extra=point_errors(val,'validation');files+=extra
    noise=json.loads(Path('artifacts/c_q2/protocol/training_noise.json').read_text())
    folds=json.loads(Path('artifacts/c_q2/protocol/folds.json').read_text())
    counts=pd.read_csv('artifacts/c_q2/protocol/development_windows.csv').set_index('window_id').target_times*2
    for p in (train,val):
        p['meta']['expected_hand_times']=p['meta'].window_id.map(counts)
    scale=train['meta'].dataset.map({d:v['hand_scale'] for d,v in noise.items()}).to_numpy()
    vscale=val['meta'].dataset.map({d:v['hand_scale'] for d,v in noise.items()}).to_numpy()
    x=transform_features(train['features'],train['feature_names'],scale);vx=transform_features(val['features'],val['feature_names'],vscale)
    y=train_error/scale;vy=val_error/vscale;age=train['meta'].age_s.to_numpy();vage=val['meta'].age_s.to_numpy()
    files += [Path(__file__),Path('src/shuwei_c/q2_radius_refine.py'),Path('src/shuwei_c/q2_calibration.py'),
              Path('src/shuwei_c/q2_features.py'),Path('src/shuwei_c/q2_learning.py'),
              Path('scripts/analyze_c_q2_calibration.py'),Path('artifacts/c_q2/protocol/training_noise.json'),
              Path('artifacts/c_q2/protocol/folds.json'),Path('artifacts/c_q2/protocol/development_windows.csv')]
    for split in ('train','validation'):
        files.extend(sorted(Path('artifacts/c_q2/learning_data',split).glob('*.npz')))
        files.extend(sorted(Path('artifacts/c_q2/learning_data',split).glob('*.parquet')))
        files.append(Path('artifacts/c_q2/learning_data',split,'manifest.json'))
    parameters=dict(n_estimators=150,num_leaves=15,min_child_samples=100,learning_rate=.04,reg_lambda=10.,n_jobs=2)
    identity=dict(point='equal-weight static/residual ensembles',levels=args.levels,fit_per_group=args.fit_per_group,
                  parameters=parameters,test_used=False,validation_used_for_fitting=False,
                  whole_pipeline_oof=False,correction='age-conditioned training cross-fitted error/radius quantile',
                  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    ip=args.out/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Changed calibration inputs')
    ip.write_text(json.dumps(identity,indent=2)+'\n')
    for p in files:
        if p.suffix!='.py':continue
        relative=p.relative_to(Path.cwd()) if p.is_absolute() else p
        dst=args.out/'source_snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    started=time.perf_counter();oof=np.full((len(y),len(args.levels)),np.nan)
    for fold in folds:
        folder=args.out/f'calibration_fold{fold["fold"]}';folder.mkdir(exist_ok=True)
        held=np.flatnonzero(train['meta'].environment.isin(fold['held_environments']).to_numpy())
        fitting=sample(train['meta'],fold['training_environments'],args.fit_per_group)
        if (folder/'complete.json').exists():
            oof[held]=np.load(folder/'held_radii.npy');continue
        model=RadiusCalibrator(levels=args.levels,parameters=parameters).fit(x[fitting],y[fitting],train['meta'].iloc[fitting])
        radius=model.predict(x[held]);radius[age[held]<=1e-9]=0;oof[held]=radius
        model.save(folder/'models');np.save(folder/'held_radii.npy',radius)
        np.savez_compressed(folder/'indices.npz',fitting=fitting,held=held)
        (folder/'complete.json').write_text(json.dumps(dict(rows_fit=len(fitting),rows_held=len(held),elapsed_s=time.perf_counter()-started))+'\n')
        print(json.dumps(dict(stage='radius_crossfit',fold=fold['fold'],rows_fit=len(fitting),elapsed_s=time.perf_counter()-started)),flush=True)
    correction=AgeRatioCorrection(levels=args.levels).fit(age,y,oof,train['meta']);correction.save(args.out/'age_ratio.json')
    ix=sample(train['meta'],sorted(train['meta'].environment.unique()),args.fit_per_group)
    model=RadiusCalibrator(levels=args.levels,parameters=parameters).fit(x[ix],y[ix],train['meta'].iloc[ix]);model.save(args.out/'conditional')
    base=model.predict(vx);base[vage<=1e-9]=0
    age_model=AgeBinnedRadius(levels=args.levels).fit(age,y,train['meta']);age_model.save(args.out/'age_binned')
    candidates=dict(conditional=base,global_corrected=np.maximum.accumulate(np.maximum(base,1e-5)*correction.global_ratio,axis=1),
                    age_corrected=correction.predict(vage,base),age_binned=age_model.predict(vage))
    tables=[];path_tables=[];suggestion_tables=[]
    for name,radii in candidates.items():
        radii[vage<=1e-9]=0
        frame=val['meta'].copy();frame['normalized_error']=vy;frame['error']=val_error;frame['scale']=vscale
        for j,q in enumerate(args.levels):frame[f'radius_{q:g}']=radii[:,j]
        frame.to_parquet(args.out/f'{name}_radii.parquet',index=False)
        tables.append(evaluation.condition_scores(val['meta'],vy,radii,args.levels,'validation',name))
        paths=evaluate_paths_long(val['meta'],vy,radii,levels=args.levels)
        paths.to_parquet(args.out/f'{name}_paths.parquet',index=False)
        path_tables.append(evaluation.path_condition_scores(paths,'validation',name))
        suggestions=evaluation.no_request_suggestions(val['meta'],vy,radii,args.levels)
        if len(suggestions):
            suggestions.to_parquet(args.out/f'{name}_suggestions.parquet',index=False)
            suggestion_tables.append(evaluation.summarize_suggestions(suggestions,'validation',name))
    for kind,frames,fields in [('point',tables,['coverage','mean_radius','pinball']),
                               ('path',path_tables,['pointwise_coverage','entire_path_coverage','mean_radius']),
                               ('suggestion',suggestion_tables,['suggestion_rate','mean_suggested_s','selected_interval_within_tolerance'])]:
        if not frames:continue
        table=pd.concat(frames,ignore_index=True);table.to_csv(args.out/f'{kind}_environment_metrics.csv',index=False)
        keys=[c for c in ('split','dataset','horizon_s','cap','delay_s','nominal','tolerance','subset','method') if c in table]
        table.groupby(keys,observed=True,dropna=False)[fields].mean().reset_index().to_csv(args.out/f'{kind}_summary.csv',index=False)
    (args.out/'manifest.json').write_text(json.dumps(dict(**identity,elapsed_s=time.perf_counter()-started,
        status='development validation; full-pipeline outer audit and policy-distribution calibration remain'),indent=2)+'\n')


if __name__=='__main__':main()
