#!/usr/bin/env python3
"""Choose parsimonious video risk models by training environment OOF first."""
from pathlib import Path
import json,sys,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from shuwei_c import q1v2_quality as q
from shuwei_c import q1v3_shared_risk as m


def main():
    out=Path('artifacts/c_q1_v3/shared_risk');out.mkdir(exist_ok=True);start=time.perf_counter()
    train=pd.read_parquet('artifacts/c_q1_v2/quality_numerical/features_train_0.1.parquet');val=pd.read_parquet('artifacts/c_q1_v2/quality_numerical/features_validation_0.1.parquet')
    visual={s:np.load(f'artifacts/c_q1_v3/video_quality/visual_{s}.npz')['diagnostics'] for s in ['train','validation']}
    folds=list(GroupKFold(4).split(train,groups=train.environment));priors=[]
    for fi,hi in folds:
        f=train.iloc[fi];po=np.zeros((len(f),4))
        for a,b in GroupKFold(3).split(f,groups=f.environment):po[b]=q.predict_baseline(q.fit_baseline(f.iloc[a]),f.iloc[b])
        priors.append((po,q.predict_baseline(q.fit_baseline(f),train.iloc[hi])))
    full_prior=np.zeros((len(train),4))
    for (_,hi),(_,p) in zip(folds,priors):full_prior[hi]=p
    valprior=q.predict_baseline(q.fit_baseline(train),val);predictions={};rows=[];selection=[]
    for video in [False,True]:
        x,flag,names=m.features(train,visual['train'],video);vx,vflag,_=m.features(val,visual['validation'],video)
        for objective in ['brier','logloss']:
            for penalty in [.0001,.001,.01]:
                name=f'{"video" if video else "numeric"}_{objective}_{penalty}';oof=np.zeros((len(train),4));diagnostics=[]
                for (fi,hi),(fp,hp) in zip(folds,priors):
                    model=m.fit(train.iloc[fi],x[fi],flag[fi],fp,penalty,objective);oof[hi]=m.predict(x[hi],flag[hi],hp,model);diagnostics.append(dict(converged=model['converged'],iterations=model['iterations']))
                score=next(r['brier'] for r in q.metrics(train,oof,name) if r['dataset']=='ALL' and r['target']=='any' and r['condition']=='all')
                selection.append(dict(method=name,video=video,objective=objective,penalty=penalty,oof_brier=score,diagnostics=diagnostics))
                model=m.fit(train,x,flag,full_prior,penalty,objective);joblib.dump(dict(model,feature_names=names),out/(name+'.joblib'))
                pred=m.predict(vx,vflag,valprior,model);predictions[name+'_validation']=pred;predictions[name+'_oof']=oof
                for split,frame,p in [('train_oof',train,oof),('validation',val,pred)]:
                    values=q.metrics(frame,p,name)
                    for row in values:row['split']=split
                    rows+=values
                pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);print(json.dumps(dict(method=name,oof_brier=score,elapsed_s=time.perf_counter()-start)),flush=True)
    np.savez_compressed(out/'probabilities.npz',**predictions);(out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,elapsed_s=time.perf_counter()-start,selection='Use environment-OOF Brier before inspecting validation ranking',source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_shared_risk.py')]}),indent=2)+'\n')


if __name__=='__main__':main()
