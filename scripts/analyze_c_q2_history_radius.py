#!/usr/bin/env python3
"""History-gap quantiles and joint-path correction on development data only."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q2_learning import load_pack
from shuwei_c.q2_history_radius import HistoryBinnedRadius,scale_path_radius
from shuwei_c.q2_calibration import hierarchical_weights,weighted_quantile,evaluate_paths_long


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def path_factors(meta,error,radii,levels):
    take=meta.cap.eq(0).to_numpy();f=meta.loc[take].copy()
    keys=['environment','episode','dataset','window_id','scenario','horizon_s','phase']
    for j,q in enumerate(levels):f[f'ratio{j}']=error[take]/np.maximum(radii[take,j],1e-5)
    paths=f.groupby(keys,sort=False)[[f'ratio{j}' for j in range(len(levels))]].max().reset_index()
    paths['hand']=0;factors={}
    for horizon,group in paths.groupby('horizon_s',sort=True):
        w=hierarchical_weights(group)
        factors[str(horizon)]=[weighted_quantile(group[f'ratio{j}'].to_numpy(),q,w) for j,q in enumerate(levels)]
    return factors,paths


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('artifacts/c_q2/calibration/history_gap'))
    parser.add_argument('--reuse-fits',type=Path)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    selected=module('selected_radius','scripts/calibrate_c_q2_selected.py')
    evaluation=module('radius_evaluation','scripts/analyze_c_q2_calibration.py')
    train=load_pack('artifacts/c_q2/learning_data/train');val=load_pack('artifacts/c_q2/learning_data/validation')
    e,files=selected.point_errors(train,'train');ve,more=selected.point_errors(val,'validation');files+=more
    noise=json.loads(Path('artifacts/c_q2/protocol/training_noise.json').read_text())
    folds=json.loads(Path('artifacts/c_q2/protocol/folds.json').read_text());levels=[.5,.8,.9,.95]
    counts=pd.read_csv('artifacts/c_q2/protocol/development_windows.csv').set_index('window_id').target_times*2
    def values(pack,error):
        meta=pack['meta'];meta['expected_hand_times']=meta.window_id.map(counts)
        scale=meta.dataset.map({d:n['hand_scale'] for d,n in noise.items()}).to_numpy()
        names=pack['feature_names']
        return meta.age_s.to_numpy(),pack['features'][:,names.index('latest_gap_s')],pack['features'][:,names.index('speed_0')]/scale,error/scale,scale
    age,gap,speed,error,scale=values(train,e);va,vg,vs,verror,vscale=values(val,ve)
    files += [Path(__file__),Path('scripts/calibrate_c_q2_selected.py'),Path('scripts/analyze_c_q2_calibration.py'),
              Path('src/shuwei_c/q2_history_radius.py'),Path('src/shuwei_c/q2_calibration.py'),Path('src/shuwei_c/q2_learning.py'),
              Path('artifacts/c_q2/protocol/development_windows.csv'),Path('artifacts/c_q2/protocol/training_noise.json'),Path('artifacts/c_q2/protocol/folds.json')]
    for split in ('train','validation'):
        files+=list(Path('artifacts/c_q2/learning_data',split).glob('*.npz'))+list(Path('artifacts/c_q2/learning_data',split).glob('*.parquet'))
        files.append(Path('artifacts/c_q2/learning_data',split,'manifest.json'))
    identity=dict(source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  levels=levels,point='fixed equal static/residual blend',whole_pipeline_oof=False,
                  split_use='training crossfit correction; official validation excludes all fitting',test_used=False,
                  variants=['age_gap','age_gap_rate','age_gap_motion','age_gap_rate_motion'])
    if args.reuse_fits:
        parent=json.loads((args.reuse_fits/'identity.json').read_text())
        for key,value in parent['source_hashes'].items():
            if not key.endswith('.py') and identity['source_hashes'].get(key)!=value:
                raise ValueError(f'Reused fitting inputs changed: {key}')
        for key in ('point','levels','variants'):
            if identity[key]!=parent[key]:raise ValueError(f'Reused specification changed: {key}')
        reused=[args.reuse_fits/'identity.json']+list(args.reuse_fits.glob('*/fold*.json'))+list(args.reuse_fits.glob('*/full.json'))+list(args.reuse_fits.glob('*/training_crossfit_radii.npy'))
        identity['reused_fit_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in reused}
        identity['reuse_reason']='Same point/data/fit specification; add load and match path-application floor to path-score floor. Preserve original incomplete run.'
    ip=out/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Inputs changed')
    ip.write_text(json.dumps(identity,indent=2)+'\n')
    for p in files:
        if p.suffix!='.py':continue
        relative=p.relative_to(Path.cwd()) if p.is_absolute() else p
        dst=out/'source_snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    started=time.perf_counter();points=[];paths=[];suggestions=[]
    for rate,motion in ((False,False),(True,False),(False,True),(True,True)):
        name='age_gap'+('_rate' if rate else '')+('_motion' if motion else '')
        folder=out/name;folder.mkdir(exist_ok=True);oof=np.zeros((len(error),len(levels)))
        for fold in folds:
            fit=train['meta'].environment.isin(fold['training_environments']).to_numpy();held=~fit
            prior=args.reuse_fits/name/f'fold{fold["fold"]}.json' if args.reuse_fits else None
            model=HistoryBinnedRadius.load(prior) if prior and prior.exists() else HistoryBinnedRadius(levels,rate,motion).fit(age[fit],gap[fit],speed[fit],error[fit],train['meta'].loc[fit])
            if set(model.fit_environments)!=set(train['meta'].loc[fit,'environment'].unique()):raise ValueError('Reused fold mismatch')
            oof[held]=model.predict(age[held],gap[held],speed[held]);model.save(folder/f'fold{fold["fold"]}.json')
        factors,score=path_factors(train['meta'],error,oof,levels)
        score.to_parquet(folder/'training_crossfit_path_scores.parquet',index=False)
        (folder/'path_factors.json').write_text(json.dumps(factors,indent=2)+'\n')
        prior=args.reuse_fits/name/'full.json' if args.reuse_fits else None
        model=HistoryBinnedRadius.load(prior) if prior and prior.exists() else HistoryBinnedRadius(levels,rate,motion).fit(age,gap,speed,error,train['meta'])
        if set(model.fit_environments)!=set(train['meta'].environment.unique()):raise ValueError('Reused full fit mismatch')
        model.save(folder/'full.json')
        radius=model.predict(va,vg,vs);np.save(folder/'training_crossfit_radii.npy',oof)
        for mode in ('point','path'):
            meta=val['meta'].copy();y=verror.copy();r=radius.copy()
            if mode=='path':
                take=meta.cap.eq(0).to_numpy();meta=meta.loc[take].copy();y=y[take];r=r[take]
                for horizon,group in meta.groupby('horizon_s',sort=False):
                    mask=meta.horizon_s.eq(horizon).to_numpy();r[mask]=scale_path_radius(r[mask],factors[str(horizon)],meta.loc[mask,'age_s'])
                r=np.maximum.accumulate(r,axis=1)
            tag=name+('_joint_path' if mode=='path' else '')
            frame=meta.copy();frame['normalized_error']=y
            for j,q in enumerate(levels):frame[f'radius_{q:g}']=r[:,j]
            frame.to_parquet(folder/f'{mode}_validation_radii.parquet',index=False)
            points.append(evaluation.condition_scores(meta,y,r,levels,'validation',tag))
            path=evaluate_paths_long(meta,y,r,levels=levels);path.to_parquet(folder/f'{mode}_validation_paths.parquet',index=False)
            paths.append(evaluation.path_condition_scores(path,'validation',tag))
            suggestion=evaluation.no_request_suggestions(meta,y,r,levels)
            suggestion.to_parquet(folder/f'{mode}_validation_suggestions.parquet',index=False)
            suggestions.append(evaluation.summarize_suggestions(suggestion,'validation',tag))
        print(json.dumps(dict(method=name,elapsed_s=time.perf_counter()-started)),flush=True)
    for kind,frames,fields in [('point',points,['coverage','mean_radius','pinball']),
                               ('path',paths,['pointwise_coverage','entire_path_coverage','mean_radius']),
                               ('suggestion',suggestions,['suggestion_rate','mean_suggested_s','selected_interval_within_tolerance'])]:
        table=pd.concat(frames,ignore_index=True);table.to_csv(out/f'{kind}_environment_metrics.csv',index=False)
        keys=[c for c in ('split','dataset','horizon_s','cap','delay_s','nominal','tolerance','subset','method') if c in table]
        table.groupby(keys,dropna=False,observed=True)[fields].mean().reset_index().to_csv(out/f'{kind}_summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(**identity,elapsed_s=time.perf_counter()-started),indent=2)+'\n')


if __name__=='__main__':main()
