#!/usr/bin/env python3
"""Fixed-budget delayed-snapshot probe using only explicitly received channels.

Training references use whole-environment OOF point models. Final validation
uses the fixed full-training point bundle. This is a declared interface probe,
not the full Q2 policy/budget benchmark or a coverage guarantee.
"""
from pathlib import Path
import copy,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
from shuwei_c.data import Dataset
from shuwei_c.motion import estimate_noise
from shuwei_c.q1v3_state import pack_records,fit_distance,row_weights,score
from shuwei_c.q1v3_learning import fit_gate
from shuwei_c.q1v3_interface import Q1V3
from shuwei_c.q1v2_state import weighted_quantile


def replay(rec,spec,fold):
    api=Q1V3(spec,rec.dataset);t,p,bad=rec.hands();centres=p.mean(axis=2);head=rec.streams['head'];rows=[];requests=[]
    for fraction in [.2,.5,.8]:
        start=int(np.searchsorted(t,t[0]+fraction*(t[-1]-t[0])))
        if start<14 or t[start]+1>t[-1]:continue
        stop=int(np.searchsorted(t,t[start]+1,side='right'))
        for interval in [.2,.5]:
            for delay in [0.,.1]:
                received=list(range(start));pending=[];next_request=t[start];known_head=set(np.flatnonzero(head['time']<t[start]-1e-9))
                for j in range(start,stop):
                    now=t[j]
                    if now+1e-9>=next_request:
                        pending.append((j,now+delay));next_request+=interval;requests.append(dict(episode=rec.episode,fraction=fraction,interval=interval,delay=delay,source_index=j,request_s=float(now),return_s=float(now+delay)))
                    ready=[entry for entry in pending if entry[1]<=now+1e-9]
                    for index,arrival in ready:
                        received.append(index);hi=np.searchsorted(head['time'],t[index],side='right')-1
                        if hi>=0:known_head.add(int(hi))
                        pending.remove((index,arrival))
                    ix=np.array(received);hx=np.array(sorted(known_head));age=float(now-t[ix[-1]]);h=min([1/30,.1,.2,.5],key=lambda v:abs(v-age))
                    position,pack=api.point(t[ix],p[ix],bad[ix],head['time'][hx],head['position'][hx],now,h)
                    if age<1e-9:position=centres[ix[-1]].copy()
                    velocity=(centres[ix[-1]]-centres[ix[-2]])/(t[ix[-1]]-t[ix[-2]])
                    predictions={'constant':centres[ix[-1]],'linear':centres[ix[-1]]+age*velocity,'gate':position}
                    for hand in range(2):
                        if bad[j,hand]:continue
                        row=dict(episode=rec.episode,environment=rec.environment,dataset=rec.dataset,split=rec.split,fold=fold,hand=hand,fraction=fraction,interval=interval,delay=delay,time_s=float(now),latest_received_s=float(t[ix[-1]]),gap_s=age,elapsed_window_s=float(now-t[start]),scale=api.scale,nominal_horizon=h)
                        for name,value in predictions.items():row[name+'_error']=float(np.linalg.norm(value[hand]-centres[j,hand]))
                        for k,name in enumerate(pack['feature_names']):row['x_'+name]=float(pack['features'][hand,k])
                        row['x_gap_s']=age;row['x_probe_interval']=interval;row['x_response_delay']=delay;row['x_window_age']=row['elapsed_window_s'];rows.append(row)
    return rows,requests


def main():
    root=Path('artifacts/c_q1_v3');out=root/'sparse_probe';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Completed sparse probe exists')
    start=time.perf_counter();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
    data=Dataset();records=data.records('train');spec=json.loads((root/'frozen_numeric/freeze.json').read_text());config=json.loads(Path('configs/c_q1_v2.json').read_text());folds=json.loads((root/'numeric_oof/folds.json').read_text());rows=[];ledger=[]
    for item in folds:
        fold=item['fold'];fr=[r for r in records if r.environment in item['training_environments']];hr=[r for r in records if r.environment in item['held_environments']];local=copy.deepcopy(spec);local['noise']=estimate_noise(fr);local['point_checkpoints']={}
        for h in [1/30,.1,.2,.5]:
            if h==.1:paths=[root/'numeric_oof'/f'fold{fold}_seed{s}.pt' for s in [20261120,20261121,20261122]]
            elif h==.5:paths=[root/'long_state_oof'/f'fold{fold}_base_s{s}.pt' for s in [20261120,20261121,20261122]]
            else:
                paths=[out/f'fold{fold}_h{h:.6f}_s{s}.pt' for s in [20261120,20261121,20261122]]
                if not all(p.exists() for p in paths):
                    pack=pack_records(fr,h,config,local['noise']);base=fit_distance(pack,.01,True)
                    for seed,path in zip([20261120,20261121,20261122],paths):torch.save(fit_gate(pack,base,10,.001,seed),path)
            local['point_checkpoints'][str(h)]=list(map(str,paths))
        for rec in hr:
            values,requests=replay(rec,local,fold);rows+=values;ledger+=requests
        print(json.dumps(dict(stage='train_oof',fold=fold,rows=len(rows),elapsed_s=time.perf_counter()-start)),flush=True)
    for rec in data.records('validation'):
        values,requests=replay(rec,spec,-1);rows+=values;ledger+=requests;print(json.dumps(dict(stage='validation',episode=rec.episode,elapsed_s=time.perf_counter()-start)),flush=True)
    allrows=pd.DataFrame(rows);allrows.to_parquet(out/'replay.parquet',index=False);pd.DataFrame(ledger).to_csv(out/'request_ledger.csv',index=False)
    train=allrows[(allrows.split=='train')&(allrows.gap_s>1e-9)].reset_index(drop=True);val=allrows[allrows.split=='validation'].reset_index(drop=True);cols=[c for c in train if c.startswith('x_')];y=train.gate_error.to_numpy()/train.scale.to_numpy();levels=[.5,.8,.9,.95];radii=[];parameters={}
    for level in levels:
        def fit(f):
            model=LGBMRegressor(objective='quantile',alpha=level,n_estimators=150,num_leaves=15,min_child_samples=100,learning_rate=.04,reg_lambda=10.,n_jobs=2,verbosity=-1,random_state=20261120)
            model.fit(f[cols],np.log1p(f.gate_error/f.scale),sample_weight=row_weights(f)*len(f));return model
        oof=np.zeros(len(train))
        for fi,hi in GroupKFold(4).split(train,groups=train.environment):oof[hi]=np.maximum(np.expm1(fit(train.iloc[fi]).predict(train.iloc[hi][cols])),1e-5)
        shift=weighted_quantile(y-oof,row_weights(train),level);model=fit(train);joblib.dump(model,out/f'quantile_{level}.joblib')
        radius=np.maximum(np.expm1(model.predict(val[cols]))+shift,1e-5);radius[val.gap_s.to_numpy()<=1e-9]=0.;radii.append(radius);parameters[str(level)]=dict(columns=cols,shift=shift)
    radius=np.maximum.accumulate(np.stack(radii,axis=1),axis=1);np.savez_compressed(out/'validation_radii.npz',radii=radius);(out/'calibration.json').write_text(json.dumps(dict(levels=levels,parameters=parameters),indent=2)+'\n')
    summary=[]
    for (interval,delay),g in val.groupby(['interval','delay']):
        for condition in ['all','unobserved_now']:
            f=g if condition=='all' else g[g.gap_s>1e-9]
            for name in ['constant','linear','gate']:summary.append(dict(interval=interval,delay=delay,condition=condition,method=name,n=len(f),mean_error=score(f,f[name+'_error'])))
            error=f.gate_error.to_numpy()/f.scale.to_numpy();w=row_weights(f)
            for k,level in enumerate(levels):
                r=radius[f.index,k];loss=np.where(error>=r,level*(error-r),(1-level)*(r-error));summary.append(dict(interval=interval,delay=delay,condition=condition,method='calibrated_gate',nominal=level,n=len(f),coverage=float(w@(error<=r)),radius=float(w@(r*f.scale)),pinball=float(w@(loss*f.scale))))
    pd.DataFrame(summary).to_csv(out/'validation_summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,elapsed_s=time.perf_counter()-start,intervals=[.2,.5],delays=[0.,.1],windows='1 s at 20/50/80 percent of each record',disclosure='Before window: complete prefix. During window: requested original hand/flag/head snapshot arrives only after latency; intermediate hidden samples and pixels are unavailable.',calibration='Training OOF point residuals; further radius crossfit used for empirical shift, not a fully nested distribution-free guarantee. Validation stays outside all fitting.',source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numeric_freeze_sha256=hashlib.sha256((root/'frozen_numeric/freeze.json').read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
