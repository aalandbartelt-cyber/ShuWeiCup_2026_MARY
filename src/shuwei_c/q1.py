"""Question 1: state construction, record/interval retention and reliability.

AI-assisted: OpenAI Codex, GPT-6 family, 2026-11-20.
Exact model release date unavailable. Human verification pending.
All future flags are targets only; retrospective stage labels are not predictors.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import average_precision_score,roc_auc_score
from .data import balanced_weights
from .motion import filter_record,predict

DATASETS=['EgoDemo','EgoPro','EgoStandard']
VIEWS=['head_left','head_right','left_wrist','right_wrist']


class WeightedScale(TransformerMixin,BaseEstimator):
    """Two-pass weighted variance avoids cancellation for constant dummy columns."""
    def fit(self,x,y=None,sample_weight=None):
        self.mean_=np.average(x,axis=0,weights=sample_weight)
        variance=np.average((x-self.mean_)**2,axis=0,weights=sample_weight)
        self.scale_=np.where(variance>1e-14,np.sqrt(np.maximum(variance,0)),1.)
        return self

    def transform(self,x):return (x-self.mean_)/self.scale_


def contiguous_runs(mask):
    changes=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))


def audit_records(records):
    rows=[];intervals=[];hashes={}
    for rec in records:
        t,p,bad=rec.hands();finite=np.isfinite(p).all(axis=(2,3));valid=finite&~bad
        dt=np.diff(t);median=float(np.median(dt));gap=dt>1.5*median
        digest=hashlib.sha256(p.tobytes()+t.tobytes()).hexdigest()
        hashes.setdefault(digest,[]).append(rec.episode)
        hand_norm=np.concatenate([np.linalg.norm(rec.streams[s]['quaternion'],axis=-1).ravel() for s in ['left','right']])
        rows.append(dict(episode=rec.episode,dataset=rec.dataset,environment=rec.environment,operator=rec.operator,
            split=rec.split,frames=len(t),duration_s=float(t[-1]-t[0]),median_dt_s=median,max_dt_s=float(dt.max()),
            gaps_over_1_5dt=int(gap.sum()),nonfinite_hand_frames=int((~finite).sum()),
            left_bad_frames=int(bad[:,0].sum()),right_bad_frames=int(bad[:,1].sum()),
            any_bad_frames=int(bad.any(axis=1).sum()),qualified_hand_frames=int(valid.sum()),total_hand_frames=2*len(t),
            consecutive_identical_hand_frames=int(np.all(np.diff(p,axis=0)==0,axis=(1,2,3)).sum()),
            quaternion_norm_max_deviation=float(np.max(np.abs(hand_norm-1))),
            quality_time_exact=bool(np.array_equal(rec.streams['quality']['time'],t)),
            head_time_exact=bool(np.array_equal(rec.streams['head']['time'],t)),
            trajectory_sha256=digest))
        for side,j in [('left',0),('right',1)]:
            for qualified in [False,True]:
                mask=valid[:,j] if qualified else ~valid[:,j]
                # Break intervals at temporal gaps, even when quality does not change.
                for start,stop in contiguous_runs(mask):
                    boundaries=[start,*list(np.flatnonzero(gap[start:stop-1])+start+1),stop]
                    for a,b in zip(boundaries[:-1],boundaries[1:]):
                        intervals.append(dict(episode=rec.episode,dataset=rec.dataset,environment=rec.environment,
                            split=rec.split,hand=side,start_index=int(a),stop_index_exclusive=int(b),
                            start_s=float(t[a]),last_sample_s=float(t[b-1]),frames=int(b-a),
                            approximate_support_s=float(t[b-1]-t[a]+median),
                            use='fit_and_reference' if qualified else 'downweighted_fit_and_low_quality_analysis',
                            error_reference_allowed=qualified))
    return pd.DataFrame(rows),pd.DataFrame(intervals),[v for v in hashes.values() if len(v)>1]


def state_evaluation(records,noise,methods=('constant','linear','filter','no_quality'),horizons=(0.,.1)):
    rows=[]
    for rec in records:
        t,p,bad=rec.hands();c=p.mean(axis=2)
        needed=([True] if 'filter' in methods else [])+([False] if 'no_quality' in methods else [])
        fitted={q:filter_record(rec,noise[rec.dataset],q) for q in needed}
        for horizon in horizons:
            origin=np.arange(2,len(t)-1)
            # horizon=0 denotes one next sample, never a posterior fitted to its own target.
            target=origin+1 if horizon==0 else np.searchsorted(t,t[origin]+horizon-1e-9)
            take=target<len(t);origin=origin[take];target=target[take]
            for method in methods:
                if method=='constant':prediction=c[origin]
                elif method=='linear':prediction=c[origin]+(c[origin]-c[origin-1])/np.diff(t)[origin-1,None,None]*(t[target]-t[origin])[:,None,None]
                else:
                    states,covs=fitted[method!='no_quality']
                    prediction=states[origin,:,0]+states[origin,:,1]*(t[target]-t[origin])[:,None,None]
                qualified=~bad[target]&np.isfinite(c[target]).all(axis=-1)
                for j,hand in enumerate(['left','right']):
                    for context in ['all','recent_flag','unflagged_history']:
                        history_bad=bad[origin,j]|bad[origin-1,j]
                        valid=qualified[:,j] & (history_bad if context=='recent_flag' else ~history_bad if context=='unflagged_history' else True)
                        if not valid.any():continue
                        errors=np.linalg.norm(prediction[valid,j]-c[target[valid],j],axis=-1)
                        rows.append(dict(episode=rec.episode,environment=rec.environment,dataset=rec.dataset,split=rec.split,
                            method=method,horizon='one_step' if horizon==0 else '0.1s',hand=hand,context=context,
                            n=len(errors),mean_euclidean_error=float(errors.mean()),p90_error=float(np.quantile(errors,.9)),
                            actual_horizon_mean_s=float(np.mean(t[target[valid]]-t[origin[valid]]))))
    return pd.DataFrame(rows)


def aggregate_state(frame):
    rows=[]
    for key,g in frame.groupby(['dataset','method','horizon','context']):
        # Hands averaged within episode first; episodes within environment then environments.
        values=g.groupby(['environment','episode']).mean_euclidean_error.mean().reset_index()
        env=values.groupby('environment').mean_euclidean_error.mean()
        rows.append(dict(zip(['dataset','method','horizon','context'],key),
            mean_euclidean_error=float(env.mean()),environments=len(env),episodes=len(values),qualified_hand_predictions=int(g.n.sum())))
    return pd.DataFrame(rows)


def load_visual(rec,root,times):
    columns=[];availability=[]
    for view in VIEWS:
        path=Path(root)/f'{rec.episode}__{view}.jsonl'
        if not path.exists():
            columns.append(np.zeros((len(times),8)));availability.append(np.zeros(len(times),bool));continue
        raw=[json.loads(line) for line in path.open()]
        vt=np.array([r['time_s'] for r in raw]);features=[]
        for row in raw:
            detections=row['detections'];n=len(detections)
            if n:
                landmarks=np.array([r['landmarks'] for r in detections])[:,:,:2]
                centres=landmarks.mean(axis=1)
                features.append([1.,float(n),*centres.mean(axis=0),float(np.std(centres[:,0])),
                    float(np.mean(np.prod(np.ptp(landmarks,axis=1),axis=1))),
                    float(np.mean([r['handedness_score'] for r in detections])),0.])
            else:features.append([1.,0.,0.,0.,0.,0.,0.,0.])
        ix=np.searchsorted(vt,times,side='right')-1;valid=ix>=0
        f=np.zeros((len(times),8));f[valid]=np.asarray(features)[ix[valid]]
        age=np.where(valid,times-vt[np.maximum(ix,0)],np.inf)
        # Only images at or before the current timestamp, at most 0.25 s stale.
        valid &= age<=.25
        f[~valid]=0.;f[valid,-1]=age[valid]
        columns.append(f);availability.append(valid)
    return np.column_stack(columns),np.column_stack(availability)


def reliability_data(records,video_root=None):
    xs=[];ys=[];meta=[]
    for rec in records:
        t,p,bad=rec.hands();c=p.mean(axis=2);origin=np.arange(2,len(t)-1)
        target=np.searchsorted(t,t[origin]+.1-1e-9);take=target<len(t);origin=origin[take];target=target[take]
        velocity=(c[origin]-c[origin-1])/(t[origin]-t[origin-1])[:,None,None]
        previous=(c[origin-1]-c[origin-2])/(t[origin-1]-t[origin-2])[:,None,None]
        radius=np.linalg.norm(p[origin]-c[origin,:,None,:],axis=-1).mean(axis=-1)
        features=np.column_stack([np.log1p(np.linalg.norm(velocity,axis=-1)),
            np.log1p(np.linalg.norm(velocity-previous,axis=-1)),radius,
            np.linalg.norm(c[origin,0]-c[origin,1],axis=-1),bad[origin].astype(float),
            np.tile([rec.dataset==d for d in DATASETS],(len(origin),1))])
        if video_root:
            visual,available=load_visual(rec,video_root,t[origin]);features=np.column_stack([features,visual])
        else:available=np.zeros((len(origin),4),bool)
        valid=np.isfinite(features).all(axis=1)
        xs.append(features[valid]);ys.append(bad[target[valid]].any(axis=1).astype(int))
        for i,j,a in zip(origin[valid],target[valid],available[valid]):
            meta.append(dict(episode=rec.episode,environment=rec.environment,operator=rec.operator,dataset=rec.dataset,
                time_s=float(t[i]),target_time_s=float(t[j]),current_bad=int(bad[i].any()),
                available_views=int(a.sum()),history_speed=float(np.linalg.norm((c[i]-c[i-1])/(t[i]-t[i-1]),axis=-1).mean())))
    return np.vstack(xs),np.concatenate(ys),pd.DataFrame(meta)


def model_columns(variant,n):
    if variant=='motion_only':return [*range(7),*range(9,12)]
    if variant=='numeric':return list(range(12))
    if variant=='multimodal':return list(range(n))
    raise ValueError(variant)


def fit_classifier(x,y,weights,kind):
    if kind=='logistic':
        model=Pipeline([('scale',WeightedScale()),('classifier',LogisticRegression(C=.1,max_iter=1000,random_state=20261120))])
        model.fit(x,y,scale__sample_weight=weights*len(weights),classifier__sample_weight=weights*len(weights))
    else:
        model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=80,l2_regularization=10.,
            learning_rate=.06,early_stopping=False,random_state=20261120)
        model.fit(x,y,sample_weight=weights*len(weights))
    return model


def weights(meta):return balanced_weights(meta.environment,meta.episode)


def reliability_metrics(y,prob,meta,method):
    rows=[]
    for ds in [*DATASETS,'ALL']:
        take=np.ones(len(y),bool) if ds=='ALL' else (meta.dataset==ds).to_numpy()
        if not take.any():continue
        yy,pp=y[take],np.clip(prob[take],1e-7,1-1e-7);w=weights(meta[take])
        ece=0.
        for low,high in zip([0,.01,.05,.1,.2,.4,.6,.8],[.01,.05,.1,.2,.4,.6,.8,1.00001]):
            band=(pp>=low)&(pp<high)
            if band.any():ece+=w[band].sum()*abs(np.average(pp[band]-yy[band],weights=w[band]))
        rows.append(dict(dataset=ds,method=method,frames=len(yy),positive_frames=int(yy.sum()),
            environments=meta[take].environment.nunique(),weighted_prevalence=float(np.average(yy,weights=w)),
            brier=float(np.average((pp-yy)**2,weights=w)),
            log_loss=float(np.average(-yy*np.log(pp)-(1-yy)*np.log1p(-pp),weights=w)),
            ece=float(ece),average_precision=float(average_precision_score(yy,pp,sample_weight=w)) if yy.any() else None,
            auc=float(roc_auc_score(yy,pp,sample_weight=w)) if len(np.unique(yy))==2 else None))
    return rows


def baseline_predictions(y,train_meta,eval_meta,markov=False):
    w=weights(train_meta);out=np.empty(len(eval_meta));overall=float(np.average(y,weights=w))
    for ds in DATASETS:
        for status in ([0,1] if markov else [-1]):
            take=(train_meta.dataset==ds).to_numpy()
            apply=(eval_meta.dataset==ds).to_numpy()
            if markov:take&=(train_meta.current_bad==status).to_numpy();apply&=(eval_meta.current_bad==status).to_numpy()
            # One effective-record prior pseudo-count, preventing an empirical 0/1 boundary.
            if take.any():
                eff=w[take]*len(y);rate=(float(np.dot(eff,y[take]))+overall)/(eff.sum()+1.)
            else:rate=overall
            out[apply]=rate
    return out


def fit_reliability(train,validation,output,video_root=None):
    import joblib
    output=Path(output);x,y,meta=reliability_data(train,video_root);xv,yv,mv=reliability_data(validation,video_root)
    meta.to_csv(output/'quality_training_rows.csv',index=False)
    predictions=mv.copy();predictions['target_bad']=yv;metrics=[];oof_summary=[]
    for name,markov in [('dataset_rate',False),('flag_transition',True)]:
        prob=baseline_predictions(y,meta,mv,markov);predictions[name]=prob;metrics+=reliability_metrics(yv,prob,mv,name)
    public_grid=pd.DataFrame([dict(dataset=ds,current_bad=status) for ds in DATASETS for status in [0,1]])
    public_grid['probability']=baseline_predictions(y,meta,public_grid,True)
    public_grid.to_csv(output/'flag_transition_model.csv',index=False)
    # Parameter uncertainty, preserving within-record dependence.
    rng=np.random.default_rng(20261120);envs=meta.environment.unique();base_w=weights(meta)
    cell_mass=np.zeros((len(envs),len(public_grid)));cell_positive=cell_mass.copy();env_positive=[]
    for i,env in enumerate(envs):
        env_take=(meta.environment==env).to_numpy();env_positive.append(float(np.dot(base_w[env_take],y[env_take])))
        for j,row in enumerate(public_grid.itertuples(index=False)):
            take=env_take&((meta.dataset==row.dataset)&(meta.current_bad==row.current_bad)).to_numpy()
            cell_mass[i,j]=base_w[take].sum()*len(y);cell_positive[i,j]=np.dot(base_w[take],y[take])*len(y)
    counts=rng.multinomial(len(envs),np.full(len(envs),1/len(envs)),size=2000)
    mass=counts@cell_mass;positive=counts@cell_positive;overall=counts@np.array(env_positive)
    boot=np.where(mass>0,(positive+overall[:,None])/(mass+1.),np.nan)
    public_grid['low']=np.nanquantile(boot,.025,axis=0);public_grid['high']=np.nanquantile(boot,.975,axis=0)
    public_grid['bootstrap_replicates_with_support']=np.isfinite(boot).sum(axis=0)
    public_grid.to_csv(output/'flag_transition_parameter_uncertainty.csv',index=False)
    variants=['motion_only','numeric']+(['multimodal'] if video_root else [])
    fitted={}
    for variant in variants:
        columns=model_columns(variant,x.shape[1])
        for kind in ['logistic','boosting']:
            name=variant+'_'+kind;oof=np.zeros(len(y))
            for env in sorted(meta.environment.unique()):
                held=(meta.environment==env).to_numpy()
                model=fit_classifier(x[~held][:,columns],y[~held],weights(meta[~held]),kind)
                oof[held]=model.predict_proba(x[held][:,columns])[:,1]
            # Calibration trained exclusively on grouped out-of-fold TRAIN predictions.
            calibrator=LogisticRegression(C=10.,max_iter=1000)
            calibrator.fit(logit(np.clip(oof,1e-6,1-1e-6))[:,None],y,sample_weight=weights(meta)*len(y))
            model=fit_classifier(x[:,columns],y,weights(meta),kind)
            raw=model.predict_proba(xv[:,columns])[:,1]
            prob=calibrator.predict_proba(logit(np.clip(raw,1e-6,1-1e-6))[:,None])[:,1]
            for suffix,values in [('_raw',raw),('',prob)]:
                predictions[name+suffix]=values;metrics+=reliability_metrics(yv,values,mv,name+suffix)
            oof_summary+=reliability_metrics(y,oof,meta,name+'_raw_oof')
            fitted[name]=(model,calibrator,columns)
            print(json.dumps(dict(q1_reliability=name,validation_brier=next(r['brier'] for r in metrics if r['dataset']=='ALL' and r['method']==name))),flush=True)
    result=pd.DataFrame(metrics);result.to_csv(output/'quality_validation_metrics.csv',index=False)
    pd.DataFrame(oof_summary).to_csv(output/'quality_train_oof_metrics.csv',index=False)
    predictions.to_csv(output/'quality_validation_predictions.csv',index=False)
    # Validation selection is explicit. No test performance is inferred.
    candidates=result[result.dataset=='ALL'].sort_values(['brier','log_loss'])
    selected=str(candidates.iloc[0].method)
    selection=dict(selected=selected,criterion='validation environment-balanced Brier; log-loss tie-break',
        test_used=False,calibration='Platt fit on leave-environment-out training predictions',
        models=list(fitted),selected_from=len(candidates))
    (output/'quality_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    joblib.dump(dict(fitted=fitted,selection=selection),output/'quality_models.joblib')
    # Baseline-paired environment bootstrap, with fixed fitted predictions.
    pairs=[];rng=np.random.default_rng(20261120)
    for name in predictions.columns:
        if name not in result.method.unique():continue
        delta=(predictions[name].to_numpy()-yv)**2-(predictions.flag_transition.to_numpy()-yv)**2
        env_d=[]
        for env in mv.environment.unique():
            take=(mv.environment==env).to_numpy();env_d.append(float(np.average(delta[take],weights=weights(mv[take]))))
        boot=np.mean(rng.choice(env_d,size=(4000,len(env_d))),axis=1)
        pairs.append(dict(method=name,mean_brier_difference=float(np.mean(env_d)),
            low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),independent_environments=len(env_d)))
    pd.DataFrame(pairs).to_csv(output/'quality_paired_vs_transition.csv',index=False)
    # Stress test uses only original TRAIN episodes, held out by operator.
    # Report uncalibrated refits: the full-training calibrator must not leak into a held operator.
    stress=[]
    for operator in sorted(meta.operator.unique()):
        held=(meta.operator==operator).to_numpy()
        for name,markov in [('dataset_rate',False),('flag_transition',True)]:
            prob=baseline_predictions(y[~held],meta[~held],meta[held],markov)
            stress.extend([dict(held_operator=operator,**r) for r in reliability_metrics(y[held],prob,meta[held],name)])
        for variant in ['numeric']+(['multimodal'] if video_root else []):
            columns=model_columns(variant,x.shape[1]);kind='logistic'
            model=fit_classifier(x[~held][:,columns],y[~held],weights(meta[~held]),kind)
            prob=model.predict_proba(x[held][:,columns])[:,1]
            stress.extend([dict(held_operator=operator,**r) for r in reliability_metrics(y[held],prob,meta[held],variant+'_logistic_raw')])
    pd.DataFrame(stress).to_csv(output/'quality_leave_operator_stress.csv',index=False)
    return result,predictions,selection


def retrospective_segments(data,records):
    by_episode={r.episode:r for r in records};rows=[]
    for u in data.units:
        if u['episode_uuid'] not in by_episode:continue
        rec=by_episode[u['episode_uuid']];t,p,bad=rec.hands();c=p.mean(axis=2)
        speed=np.r_[np.zeros((1,2)),np.linalg.norm(np.diff(c,axis=0),axis=-1)/np.diff(t)[:,None]]
        start,end=float(u['start_s']),float(u['end_s'])
        for k,stage in enumerate(['early','middle','late']):
            a=start+(end-start)*k/3;b=start+(end-start)*(k+1)/3
            take=(t>=a)&(t<b)
            if not take.any():continue
            clean=~bad[take]
            rows.append(dict(dataset=rec.dataset,environment=rec.environment,episode=rec.episode,split=rec.split,
                family=u['family_id'],stage=stage,frames=int(take.sum()),any_bad_fraction=float(bad[take].any(axis=1).mean()),
                clean_speed=float(speed[take][clean].mean()) if clean.any() else None,
                interhand_distance=float(np.linalg.norm(c[take,0]-c[take,1],axis=-1).mean())))
    return pd.DataFrame(rows)
