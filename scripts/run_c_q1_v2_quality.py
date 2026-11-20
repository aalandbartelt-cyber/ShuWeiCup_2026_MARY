#!/usr/bin/env python3
"""Coherent joint-state reliability, grouped calibration and event diagnostics.

AI-assisted by Codex, 2026-11-20. Train/validation development only.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from shuwei_c.data import Dataset
from shuwei_c import q1v2_quality as q


def save(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def baseline_oof(frame,transition):
    out=np.empty((len(frame),4));folds=[]
    for fold,(fit,held) in enumerate(GroupKFold(4).split(frame,groups=frame.environment)):
        out[held]=q.predict_baseline(q.fit_baseline(frame.iloc[fit],transition),frame.iloc[held])
        folds.append(dict(fold=fold,training_environments=sorted(frame.iloc[fit].environment.unique()),held_environments=sorted(frame.iloc[held].environment.unique())))
    return out,folds


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--video-root');parser.add_argument('--tag',default='numerical');args=parser.parse_args()
    started=time.perf_counter();out=Path('artifacts/c_q1_v2')/('quality_'+args.tag);out.mkdir(exist_ok=True)
    config=json.loads(Path('configs/c_q1_v2.json').read_text());data=Dataset();train=data.records('train');val=data.records('validation')
    if args.video_root:
        manifest=json.loads((Path(args.video_root)/'video_manifest.json').read_text())
        assert len(manifest)==166 and {r['split'] for r in manifest}=={'train','validation'}
        variants=['head_video','all_video']
    else:variants=['duration','motion_no_flags','motion','geometry']
    all_metrics=[];event_rows=[];event_details=[];selection=[];models={}
    # The two reliability horizons were fixed before fitting: main 0.1 s and
    # 0.5 s to expose duration/recovery behavior beyond one short bad run.
    save(out/'protocol.json',dict(horizons=[.1,.5],variants=variants,classifiers=['logistic','boost'],folds=4,
        calibration='joint-state temperature, training OOF only; validation selects raw/calibrated',
        selection='environment/episode weighted any-hand Brier',event_thresholds=config['quality_false_alarm_levels'],test_used=False))
    for h in [.1,.5]:
        training=q.dataset_frame(train,h,config,args.video_root);validation=q.dataset_frame(val,h,config,args.video_root)
        training.to_parquet(out/f'features_train_{h:.1f}.parquet',index=False);validation.to_parquet(out/f'features_validation_{h:.1f}.parquet',index=False)
        candidates=[]
        if not args.video_root:
            for transition,name in [(False,'frequency'),(True,'transition')]:
                model=q.fit_baseline(training,transition);p=q.predict_baseline(model,validation);oof,folds=baseline_oof(training,transition)
                candidates.append((name,model,None,p,oof,folds,'baseline'))
        for variant in variants:
            cols=q.columns_for(training,variant);fitted=[]
            for kind in ['logistic','boost']:
                model=q.fit_classifier(training,cols,kind);p=q.predict_classifier(model,validation,cols)
                score=float(np.dot(q.weights(validation),(1-p[:,0]-(validation.target_state>0))**2))
                selection.append(dict(horizon=h,variant=variant,kind=kind,score=score))
                fitted.append((score,kind,model,p))
            _,kind,model,p=min(fitted,key=lambda r:r[0]);oof,folds=q.grouped_oof(training,cols,kind)
            candidates.append((variant,model,cols,p,oof,folds,kind))
        models[str(h)]={}
        for name,model,cols,p,oof,folds,kind in candidates:
            temp=q.fit_temperature(oof,training)
            options=[('raw',1.,p,oof),('temperature',temp,q.temperature(p,temp),q.temperature(oof,temp))]
            scored=[]
            for calibration,T,pv,po in options:
                metric=q.metrics(validation,pv,name+'_'+calibration)
                for row in metric:row.update(horizon=h)
                all_metrics+=metric
                score=next(r['brier'] for r in metric if r['dataset']=='ALL' and r['condition']=='all' and r['target']=='any')
                scored.append((score,T,pv,po,calibration))
            score,T,pv,po,calibration=min(scored,key=lambda x:x[0])
            spec=dict(kind=kind,columns=cols,temperature=T,calibration=calibration,score=score,thresholds={},folds=folds)
            joblib.dump(model,out/f'model_{h:.1f}_{name}.joblib')
            for split,frame,probs in [('validation',validation,pv),('oof',training,po)]:
                prediction=frame[['episode','environment','dataset','origin','target','time_s','target_time_s','current_state','target_state']].copy()
                for j in range(4):prediction[f'p{j}']=probs[:,j]
                prediction.to_parquet(out/f'{split}_{h:.1f}_{name}.parquet',index=False)
            for mode in ['onset','recovery']:
                spec['thresholds'][mode]={}
                for level in config['quality_false_alarm_levels']:
                    threshold=q.select_threshold(po,training,level,mode);spec['thresholds'][mode][str(level)]=threshold
                    values,details=q.event_metrics(val,validation,pv,threshold,h,mode)
                    row=dict(horizon=h,method=name,calibration=calibration,mode=mode,nominal_false_alarm=level,threshold=threshold,**values)
                    event_rows.append(row)
                    details['horizon']=h;details['method']=name;details['nominal_false_alarm']=level;event_details.append(details)
            models[str(h)][name]=spec
            print(json.dumps(dict(horizon=h,method=name,kind=kind,calibration=calibration,brier=score)),flush=True)
        save(out/'models.json',models)
    pd.DataFrame(all_metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(selection).to_csv(out/'selection.csv',index=False)
    pd.DataFrame(event_rows).to_csv(out/'events.csv',index=False);pd.concat(event_details,ignore_index=True).to_csv(out/'event_details.csv',index=False)
    paths=[Path(__file__),Path('src/shuwei_c/q1v2_quality.py'),out/'protocol.json']
    save(out/'run_manifest.json',dict(elapsed_s=time.perf_counter()-started,test_used=False,video_root=args.video_root,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))


if __name__=='__main__':main()
