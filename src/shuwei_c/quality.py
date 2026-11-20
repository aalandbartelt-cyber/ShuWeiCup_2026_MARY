"""Forecast a future annotation flag, not physical 3D accuracy or task success.

AI: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from .data import balanced_weights


def quality_rows(records):
    x,y,meta=[],[],[]
    datasets=['EgoDemo','EgoPro','EgoStandard']
    for rec in records:
        t,p,bad=rec.hands();c=p.mean(axis=2)
        for i in range(2,len(t)-1):
            j=int(np.searchsorted(t,t[i]+.1))
            if j>=len(t):continue
            v=(c[i]-c[i-1])/(t[i]-t[i-1]);v0=(c[i-1]-c[i-2])/(t[i-1]-t[i-2])
            radius=np.linalg.norm(p[i]-c[i,:,None,:],axis=-1).mean(axis=-1)
            features=np.r_[np.log1p(np.linalg.norm(v,axis=-1)),np.log1p(np.linalg.norm(v-v0,axis=-1)),
                           radius,bad[i].astype(float),[rec.dataset==d for d in datasets]]
            if not np.isfinite(features).all():continue
            x.append(features);y.append(int(bad[j].any()))
            meta.append(dict(episode=rec.episode,environment=rec.environment,operator=rec.operator,dataset=rec.dataset,time_s=float(t[i])))
    return np.array(x),np.array(y),meta


def train_quality(records):
    x,y,meta=quality_rows(records)
    w=balanced_weights([r['environment'] for r in meta],[r['episode'] for r in meta])
    model=LogisticRegression(C=1.,max_iter=2000)
    model.fit(x,y,sample_weight=w*len(w))
    rates={}
    for ds in sorted({r['dataset'] for r in meta}):
        take=np.array([r['dataset']==ds for r in meta]);rates[ds]=float(np.average(y[take],weights=w[take]))
    return model,dict(coef=model.coef_.tolist(),intercept=model.intercept_.tolist(),
                      baseline_rate=float(np.average(y,weights=w)),baseline_rates=rates,target='any hand bad flag 0.1 s ahead',
                      features='log speed[2], log velocity change[2], radius[2], current bad[2], dataset[3]')


def evaluate_quality(model,params,records):
    x,y,meta=quality_rows(records);prob=model.predict_proba(x)[:,1]
    rows=[]
    for ds in sorted({r['dataset'] for r in meta}):
        take=np.array([r['dataset']==ds for r in meta]);m=[r for r in meta if r['dataset']==ds]
        w=balanced_weights([r['environment'] for r in m],[r['episode'] for r in m])
        yy,pp=y[take],prob[take]
        rows.append(dict(dataset=ds,n=len(yy),positives=int(yy.sum()),
            brier=float(np.average((pp-yy)**2,weights=w)),
            baseline_brier=float(np.average((params['baseline_rates'].get(ds,params['baseline_rate'])-yy)**2,weights=w)),
            average_precision=float(average_precision_score(yy,pp,sample_weight=w)) if yy.any() else None,
            auc=float(roc_auc_score(yy,pp,sample_weight=w)) if len(set(yy))>1 else None))
    return rows
