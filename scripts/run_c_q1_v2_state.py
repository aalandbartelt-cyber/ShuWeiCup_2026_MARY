#!/usr/bin/env python3
"""Run matched Q1 v2 state comparisons, grouped calibration and real-prefix checks.

AI-assisted by Codex, 2026-11-20. Train/validation only; test is a later frozen run.
"""
from pathlib import Path
import hashlib
import itertools
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset
from shuwei_c.motion import estimate_noise
from shuwei_c.q1v2_state import (frame_for,linear_moments,fit_linear,prediction_for,error_rows,
    grouped_summary,selection_score,calibrate,evaluate_sets,PrefixPredictor)

OUT=Path('artifacts/c_q1_v2')


def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def evaluate(records,horizon,config,spec,noise,cache=None):
    rows=[]
    for rec in records:
        frame=frame_for(rec,horizon,config);pred=prediction_for(rec,frame,spec,noise,cache)
        rows.extend(error_rows(rec,frame,pred,spec['method'],horizon,noise[rec.dataset]['hand_scale']))
    return pd.DataFrame(rows)


def candidates(moments,config):
    yield dict(method='constant');yield dict(method='linear')
    yield dict(method='scalar_damped',coefficients=fit_linear(moments,'scalar_damped').tolist())
    for penalty in config['ridge_penalties']:
        yield dict(method='multiscale',penalty=penalty,coefficients=fit_linear(moments,'multiscale',penalty).tolist())
    for q,bad in itertools.product([1.,10.,100.],[1.,10.,100.]):
        yield dict(method='quality_filter',q_scale=q,r_scale=1.,bad_multiplier=bad)
    for cutoff,beta in itertools.product([1.,3.],[0.,1.,10.]):
        yield dict(method='one_euro',mincutoff=cutoff,beta=beta)


def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=True)
    config=json.loads(Path('configs/c_q1_v2.json').read_text());data=Dataset()
    train=data.records('train');validation=data.records('validation');noise=estimate_noise(train)
    models=dict(config_version=config['version'],noise=noise,horizons={},status='development; test unopened')
    selection=[];summaries=[];sets=[];uncertainty=[];time_rows=[];cache={};rng=np.random.default_rng(config['seed'])
    for h in config['horizons_s']:
        key=str(float(h));moments=linear_moments(train,h,config,noise);best_by_family={}
        for spec in candidates(moments,config):
            result=evaluate(validation,h,config,spec,noise,cache);score=selection_score(result)
            selection.append(dict(horizon=h,method=spec['method'],spec=json.dumps(spec,sort_keys=True),score=score))
            if spec['method'] not in best_by_family or score<best_by_family[spec['method']][0]:
                best_by_family[spec['method']]=(score,spec,result)
        # Stable insertion order favors simpler candidates on exact ties.
        score,spec,result=min(best_by_family.values(),key=lambda row:row[0])
        for _,candidate,frame in best_by_family.values():
            summaries.append(grouped_summary(frame))
        combined=pd.concat([v[2] for v in best_by_family.values()],ignore_index=True)
        # All comparisons must share the exact sample identity.
        identity=['episode','hand','origin','target']
        reference=combined[combined.method=='constant'][identity].sort_values(identity).to_numpy()
        for method in combined.method.unique():
            np.testing.assert_array_equal(combined[combined.method==method][identity].sort_values(identity).to_numpy(),reference)
        combined.to_parquet(OUT/f'state_validation_{h:.6f}.parquet',index=False)
        for ds,g in result.groupby('dataset'):
            time_rows.append(dict(dataset=ds,horizon=h,n=len(g),mean=float(g.actual_horizon.mean()),
                minimum=float(g.actual_horizon.min()),maximum=float(g.actual_horizon.max())))
        # Fixed-hyperparameter leave-environment fits; held-out data never estimate
        # coefficients or hand scales. Validation chooses hyperparameters legally.
        oof=[];fold_specs=[]
        for env in sorted({r.environment for r in train}):
            fitting=[r for r in train if r.environment!=env];held=[r for r in train if r.environment==env]
            fold_noise=estimate_noise(fitting);fold_spec=dict(spec)
            if spec['method'] in ['scalar_damped','multiscale']:
                fold_moments=[]
                for row in moments:
                    if row['environment']==env:continue
                    ratio=(noise[row['dataset']]['hand_scale']/fold_noise[row['dataset']]['hand_scale'])**2
                    fold_moments.append(dict(row,gram=row['gram']*ratio,target=row['target']*ratio))
                fold_spec['coefficients']=fit_linear(fold_moments,spec['method'],spec.get('penalty',0)).tolist()
            oof.append(evaluate(held,h,config,fold_spec,fold_noise))
            fold_specs.append(dict(held_environment=env,spec=fold_spec,noise=fold_noise))
        oof=pd.concat(oof,ignore_index=True);oof.to_parquet(OUT/f'state_oof_{h:.6f}.parquet',index=False)
        cal=calibrate(oof,config['coverage_levels']);set_rows=evaluate_sets(result,cal)
        for row in set_rows:row.update(horizon=h,method=spec['method'])
        sets.extend(set_rows)
        # Minimize radial pinball loss across declared levels; report coverage too.
        all_sets=pd.DataFrame(set_rows);s=all_sets[(all_sets.dataset=='ALL')&(all_sets.context=='all')]
        scheme=str(s.groupby('scheme').pinball.mean().idxmin())
        models['horizons'][key]=dict(selected=spec,score=score,calibration=cal,set_scheme=scheme,
            family_models={name:v[1] for name,v in best_by_family.items()},fold_specs=fold_specs)
        if spec['method'] in ['scalar_damped','multiscale']:
            envs=sorted({r['environment'] for r in moments})
            grams=np.array([np.mean([r['gram'] for r in moments if r['environment']==e],axis=0) for e in envs])
            targets=np.array([np.mean([r['target'] for r in moments if r['environment']==e],axis=0) for e in envs])
            draws=[]
            for _ in range(2000):
                ix=rng.integers(len(envs),size=len(envs));g=grams[ix].mean(axis=0);b=targets[ix].mean(axis=0)
                if spec['method']=='scalar_damped':coef=np.array([b[0]/g[0,0],0.,0.,0.])
                else:
                    penalty=spec['penalty'];coef=np.linalg.solve(g+penalty*np.eye(4),b+penalty*np.array([1.,0.,0.,0.]))
                draws.append(coef)
            for j,lag in enumerate(config['lags_s']):
                uncertainty.append(dict(horizon=h,lag_s=lag,coefficient=spec['coefficients'][j],
                    low=float(np.quantile(np.array(draws)[:,j],.025)),high=float(np.quantile(np.array(draws)[:,j],.975)),
                    condition='fixed selected penalty and full-training scale; environment resampling'))
        write_json(OUT/'state_models_development.json',models)
        print(json.dumps(dict(horizon=h,selected=spec,score=score,set_scheme=scheme),ensure_ascii=False),flush=True)
    pd.DataFrame(selection).to_csv(OUT/'state_selection.csv',index=False)
    pd.concat(summaries,ignore_index=True).to_csv(OUT/'state_summary.csv',index=False)
    pd.DataFrame(sets).to_csv(OUT/'state_set_validation.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(OUT/'state_parameter_uncertainty.csv',index=False)
    pd.DataFrame(time_rows).to_csv(OUT/'state_target_times.csv',index=False)
    # Integration test on actual held-out records: prefix interface = batch forecast.
    predictor=PrefixPredictor(models,config);checks=[]
    for rec in validation:
        t,p,bad=rec.hands();c=p.mean(axis=2)
        for h in config['horizons_s']:
            f=frame_for(rec,h,config);spec=models['horizons'][str(float(h))]['selected']
            for k in np.unique(np.linspace(0,len(f['i'])-1,min(4,len(f['i'])),dtype=int)):
                i,j=int(f['i'][k]),int(f['j'][k])
                # Exact declared horizon tests the corresponding interpolation knot.
                out=predictor.forecast(t[:i+1],c[:i+1],bad[:i+1],t[i]+h,rec.dataset)
                one={**f,'i':np.array([i]),'dt':np.array([h]),'x':f['x'][k:k+1]}
                expected=prediction_for(rec,one,spec,noise,cache)[0]
                difference=float(np.max(abs(out['position']-expected)))
                np.testing.assert_allclose(out['position'],expected,atol=1e-10,rtol=1e-10)
                changed=c.copy();changed[i+1:]+=12345
                out2=predictor.forecast(t[:i+1],changed[:i+1],bad[:i+1],t[i]+h,rec.dataset)
                np.testing.assert_array_equal(out['position'],out2['position'])
                checks.append(dict(episode=rec.episode,horizon=h,origin=i,max_difference=difference))
    pd.DataFrame(checks).to_csv(OUT/'real_prefix_checks.csv',index=False)
    write_json(OUT/'run_manifest_state.json',dict(elapsed_s=time.perf_counter()-started,
        test_used=False,train_records=len(train),validation_records=len(validation),
        real_prefix_checks=len(checks),source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),Path('src/shuwei_c/q1v2_state.py'),Path('configs/c_q1_v2.json')]}))
    print('Completed state v2 development and real-prefix checks',flush=True)


if __name__=='__main__':main()
