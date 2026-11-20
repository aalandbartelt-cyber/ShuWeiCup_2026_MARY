#!/usr/bin/env python3
"""Fixed-scope policy iteration with the previous deployed policy in both tails.

Round labels are generated independently inside one outer-training partition.
There is no shared model across folds and no outer-dependent hyperparameter
selection. A labels-only pilot estimates real computation before a full run.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import lightgbm as lgb
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset,balanced_weights
from shuwei_c.q2_replay import Window,NoRequests,Periodic,replay
from shuwei_c.q2_predictors import Kinematic
from shuwei_c.q2_policy import DecisionRecorder,ValueFeatures,ValuePolicy,counterfactual_pair
from shuwei_c.q2_metrics import summarize_windows


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def state_key(value):
    context,prefix=value
    return (float(context.now),tuple(prefix))


def pick_states(buckets,maximum=8):
    """Three own-policy, three periodic, two no-request slots; deduplicated fill."""
    if maximum!=8:raise ValueError('The predeclared study fixes eight decision states')
    chosen={};membership={};all_states={}
    for name,values in buckets.items():
        for value in values:
            key=state_key(value);all_states[key]=value;membership.setdefault(key,[]).append(name)
    for name,slots in [('own',3),('periodic',3),('none',2)]:
        values=buckets[name]
        if values:
            for i in np.unique(np.rint(np.linspace(0,len(values)-1,min(slots,len(values)))).astype(int)):
                value=values[i];chosen[state_key(value)]=value
    rest=[k for k in sorted(all_states) if k not in chosen]
    need=maximum-len(chosen)
    if rest and need>0:
        for i in np.unique(np.rint(np.linspace(0,len(rest)-1,min(need,len(rest)))).astype(int)):
            chosen[rest[i]]=all_states[rest[i]]
    assert len(chosen)==min(maximum,len(all_states))
    return [(chosen[k][0],chosen[k][1],membership[k]) for k in sorted(chosen)]


def feature_arrays(frame,names,noise):
    scale=frame.dataset.map({k:v['hand_scale'] for k,v in noise.items()}).to_numpy()
    x=frame[names].to_numpy(float)
    for j,name in enumerate(names):
        if '_length_' in name or '_speed_' in name:
            x[:,j]=np.log1p(np.expm1(x[:,j])/scale)
        elif name=='x_cost_weight':x[:,j]=0.
    return x.astype(np.float32),frame.error_gain.to_numpy()/scale


def weights(frame):
    columns=['environment','episode','window_id','cap','delay_s']
    counts=frame.groupby(columns,observed=True,sort=False).size().rename('n').reset_index()
    counts['weight']=balanced_weights(counts.environment,counts.episode)/counts.n
    return frame.merge(counts[columns+['weight']],on=columns,how='left',sort=False).weight.to_numpy()


def window_of(row,rec):
    return Window(row.window_id,rec.episode,rec.dataset,rec.environment,rec.split,int(row.origin_index),int(row.stop),float(row.origin_s),float(row.horizon_s))


def generate(data,windows,noise,previous,out,previous_sha,expected_names):
    out.mkdir(exist_ok=True);completed=[];started=time.perf_counter()
    for episode,group in windows.groupby('episode',sort=False):
        target=out/f'{episode}.parquet';marker=target.with_suffix('.json')
        if marker.exists():
            item=json.loads(marker.read_text())
            if item['parent_model_sha256']!=previous_sha or item['sha256']!=sha(target):raise ValueError('Iterated labels changed')
            completed.append(item);continue
        rec=data.record(episode);numerical=rec.hands();rows=[]
        scale=noise[rec.dataset]['hand_scale']
        for row in group.itertuples(index=False):
            window=window_of(row,rec)
            for cap in (1,2):
                buckets={}
                policies={'own':ValuePolicy(previous,rec.dataset,scale,0.),'periodic':Periodic(cap,.5),'none':NoRequests()}
                for name,base in policies.items():
                    recorder=DecisionRecorder(base)
                    replay(rec,window,Kinematic('damped',damping=.2),recorder,cap,.1,numerical=numerical)
                    buckets[name]=recorder.states
                raw_features=ValueFeatures(rec.dataset,1.)
                for context,prefix,explorers in pick_states(buckets):
                    x,names=raw_features(context,0.);names=['x_'+n for n in names]
                    if names!=expected_names:raise ValueError('Initial and iterated feature schemas differ')
                    label=counterfactual_pair(rec,window,context,prefix,cap,.1,1.,
                        predictor_factory=lambda:Kinematic('damped',damping=.2),
                        continuation_factory=lambda:ValuePolicy(previous,rec.dataset,scale,0.),numerical=numerical)
                    normalized,_=ValueFeatures(rec.dataset,scale)(context,0.)
                    old_gain=float(previous.predict(normalized[None],num_threads=1)[0])
                    item=dict(episode=episode,environment=rec.environment,dataset=rec.dataset,split='train',
                              window_id=window.window_id,horizon_s=window.horizon,origin_s=window.origin,
                              cap=cap,delay_s=.1,time_s=context.now,issued_prefix=json.dumps(prefix),
                              explorers=json.dumps(explorers),parent_prediction=old_gain,
                              paired_gain_normalized=label['error_gain']/scale,**label)
                    item.update(dict(zip(names,map(float,x))));rows.append(item)
        frame=pd.DataFrame(rows);temporary=target.with_suffix('.tmp.parquet');frame.to_parquet(temporary,index=False);temporary.replace(target)
        item=dict(episode=episode,rows=len(frame),windows=len(group),sha256=sha(target),parent_model_sha256=previous_sha,elapsed_s=time.perf_counter()-started)
        save_json(marker,item);completed.append(item);print(json.dumps(dict(stage='labels',**item)),flush=True)
    manifest=dict(records=len(completed),windows=sum(x['windows'] for x in completed),rows=sum(x['rows'] for x in completed),parent_model_sha256=previous_sha,elapsed_s=time.perf_counter()-started,continuation='same-partition previous ValuePolicy in both forced-action branches',predictor='last-two-qualified damped velocity tau=.2',decisions_per_window_cap=8)
    save_json(out/'manifest.json',manifest)
    return pd.concat([pd.read_parquet(out/f"{r['episode']}.parquet") for r in completed],ignore_index=True)


def evaluate(data,windows,noise,model,round_number):
    result=[];requests=[]
    for episode,group in windows.groupby('episode',sort=False):
        rec=data.record(episode);numerical=rec.hands()
        for row in group.itertuples(index=False):
            window=window_of(row,rec)
            for cap in (1,2):
                for name,policy in [('value',ValuePolicy(model,rec.dataset,noise[rec.dataset]['hand_scale'],0.)),('periodic',Periodic(cap,.5))]:
                    output=replay(rec,window,Kinematic('damped',damping=.2),policy,cap,.1,numerical=numerical)
                    metrics=dict(output['metrics']);metrics['normalized_error']=metrics['mean_error']/noise[rec.dataset]['hand_scale'];metrics['unobserved_normalized_error']=metrics['unobserved_error']/noise[rec.dataset]['hand_scale']
                    result.append(dict(window_id=window.window_id,episode=episode,environment=rec.environment,dataset=rec.dataset,split='train_oof',horizon_s=window.horizon,origin_s=window.origin,cap=cap,delay_s=.1,cost_weight=0.,policy=name,round=round_number,method='last_two_damped_tau0.2',**metrics))
                    for request in output['requests']:
                        requests.append(dict(window_id=window.window_id,episode=episode,environment=rec.environment,dataset=rec.dataset,horizon_s=window.horizon,origin_s=window.origin,cap=cap,policy=name,round=round_number,**request))
    return pd.DataFrame(result),pd.DataFrame(requests)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--initial',type=Path,default=Path('artifacts/c_q2/value_initial'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/c_q2/value_iteration/fold0'))
    parser.add_argument('--pilot-records',type=int,default=0)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    folds_path=Path('artifacts/c_q2/protocol/folds.json');fold=json.loads(folds_path.read_text())[args.fold]
    initial_folder=args.initial/f'fold{args.fold}';initial_model=initial_folder/'lambda0.txt'
    initial_meta=json.loads((initial_folder/'complete.json').read_text())
    if set(initial_meta['fit_environments'])!=set(fold['training_environments']) or set(initial_meta['held_environments'])!=set(fold['held_environments']):raise ValueError('Initial policy partition does not match')
    schema=json.loads((args.initial/'identity.json').read_text())['feature_names']
    window_path=Path('artifacts/c_q2/protocol/development_windows.csv');windows=pd.read_csv(window_path);windows=windows[(windows.split=='train')&windows.horizon_s.isin([.5,1.])]
    fitting=windows[windows.environment.isin(fold['training_environments'])];held=windows[windows.environment.isin(fold['held_environments'])]
    if args.pilot_records:
        fitting=fitting[fitting.episode.isin(fitting.episode.drop_duplicates().iloc[:args.pilot_records])]
    if set(fitting.environment)&set(held.environment):raise ValueError('Leaked policy environments')
    files=[Path(__file__),Path('src/shuwei_c/q2_policy.py'),Path('src/shuwei_c/q2_replay.py'),Path('src/shuwei_c/q2_features.py'),Path('src/shuwei_c/q2_predictors.py'),Path('src/shuwei_c/data.py'),window_path,folds_path,initial_model]
    identity=dict(fold=args.fold,rounds=args.rounds,pilot_records=args.pilot_records,fit_environments=fold['training_environments'],held_environments=fold['held_environments'],noise=fold['noise'],features=schema,horizons=[.5,1.],caps=[1,2],delay_s=.1,cost_weight=0.,decisions_per_window_cap=8,
                  sampling='own/periodic/no-request, quotas 3/3/2 then unique equispaced fill',
                  continuation='same partition previous deployed ValuePolicy after either forced action',
                  predictor='last-two-qualified damped velocity tau=.2',iterations_selected_using_outer=False,
                  solver='lightgbm.LGBMRegressor on CPU',lightgbm_version=lgb.__version__,
                  python_version=sys.version,cpu='Intel Core Ultra 9 275HX',
                  fitting_threads=2,online_prediction_threads=1,
                  model=dict(n_estimators=180,num_leaves=15,min_child_samples=50,learning_rate=.03,reg_lambda=10.,n_jobs=2,objective='regression',random_state=20261120,deterministic=True,force_col_wise=True,verbosity=-1),
                  input_hashes={str(p):sha(p) for p in files},test_used=False,official_validation_used=False)
    path=args.out/'identity.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('Policy iteration identity changed')
    save_json(path,identity)
    for p in files:
        relative=p.relative_to(Path.cwd()) if p.is_absolute() else p;dst=args.out/'source_snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    data=Dataset();previous_path=initial_model;summaries=[];started=time.perf_counter()
    if not args.pilot_records:
        folder=args.out/'round0';folder.mkdir(exist_ok=True)
        model=lgb.Booster(model_file=str(initial_model))
        if not (folder/'complete.json').exists():
            result,request=evaluate(data,held,fold['noise'],model,0);result.to_csv(folder/'windows.csv',index=False);request.to_csv(folder/'requests.csv',index=False)
            save_json(folder/'complete.json',dict(round=0,model_sha256=sha(initial_model),rows=len(result)))
        summaries.append(pd.read_csv(folder/'windows.csv'))
    for number in range(1,args.rounds+1):
        folder=args.out/f'round{number}';folder.mkdir(exist_ok=True)
        previous=lgb.Booster(model_file=str(previous_path));previous_sha=sha(previous_path)
        labels=generate(data,fitting,fold['noise'],previous,folder/'labels',previous_sha,schema)
        counts=labels.groupby(['window_id','cap']).size()
        assert counts.max()<=8 and not labels.duplicated(['window_id','cap','time_s','issued_prefix']).any()
        agreement=dict(rows=len(labels),positive_parent_fraction=float((labels.parent_prediction>0).mean()),positive_paired_fraction=float((labels.paired_gain_normalized>0).mean()),parent_positive_but_paired_negative_fraction=float(((labels.parent_prediction>0)&(labels.paired_gain_normalized<0)).mean()))
        save_json(folder/'label_diagnostics.json',agreement)
        if args.pilot_records:
            save_json(args.out/'pilot_complete.json',dict(**identity,label_rows=len(labels),windows=len(fitting),elapsed_s=time.perf_counter()-started,training_fit_performed=False,outer_evaluation_performed=False));return
        path=folder/'lambda0.txt'
        if not path.exists():
            x,y=feature_arrays(labels,schema,fold['noise']);w=weights(labels)
            model=lgb.LGBMRegressor(**identity['model']);model.fit(x,y,sample_weight=w*len(w))
            temporary=path.with_suffix('.tmp.txt');model.booster_.save_model(str(temporary));temporary.replace(path)
        model=lgb.Booster(model_file=str(path))
        if not (folder/'complete.json').exists():
            result,request=evaluate(data,held,fold['noise'],model,number);result.to_csv(folder/'windows.csv',index=False);request.to_csv(folder/'requests.csv',index=False)
            save_json(folder/'complete.json',dict(round=number,model_sha256=sha(path),parent_model_sha256=previous_sha,fit_environments=fold['training_environments'],held_environments=fold['held_environments'],label_rows=len(labels),evaluation_rows=len(result),elapsed_s=time.perf_counter()-started))
        summaries.append(pd.read_csv(folder/'windows.csv'));previous_path=path
        print(json.dumps(dict(stage='round_complete',round=number,label_rows=len(labels),elapsed_s=time.perf_counter()-started)),flush=True)
    result=pd.concat(summaries,ignore_index=True);result.to_csv(args.out/'windows.csv',index=False)
    summarize_windows(result,['round','horizon_s','cap','policy','method']).to_csv(args.out/'summary.csv',index=False)
    save_json(args.out/'manifest.json',dict(**identity,elapsed_s=time.perf_counter()-started,status='fixed three-round training-only partition policy iteration; outer results reported without selection'))


if __name__=='__main__':main()
