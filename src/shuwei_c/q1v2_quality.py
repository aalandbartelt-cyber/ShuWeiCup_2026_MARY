"""Causal joint-hand reliability features, grouped fitting and event evaluation.

AI-assisted by Codex, 2026-11-20. Four outcome states preserve coherent marginal
and any-hand probabilities; labels describe supplied quality flags, not accuracy.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import GroupKFold
from .data import balanced_weights
from .q1 import WeightedScale,contiguous_runs
from .q1v2_state import matched_targets,history_features

DATASETS=['EgoDemo','EgoPro','EgoStandard']
VIEWS=['head_left','head_right','left_wrist','right_wrist']


def status_history(time,bad):
    age=np.zeros_like(bad,dtype=float)
    for hand in range(2):
        last=float(time[0])
        for i in range(len(time)):
            if i and bad[i,hand]!=bad[i-1,hand]:last=float(time[i])
            age[i,hand]=time[i]-last
    return age


def visual_features(rec,times,root):
    result={}
    names=['available','count','mean_x','mean_y','spread_x','area','edge','handedness_score','age','speed_2d']
    for view in VIEWS:
        path=Path(root)/f'{rec.episode}__{view}.jsonl';values=np.zeros((len(times),len(names)))
        if path.exists():
            rows=[json.loads(line) for line in path.open()];vt=np.array([r['time_s'] for r in rows]);raw=[]
            for row in rows:
                detections=row['detections'];n=len(detections)
                if n:
                    p=np.array([d['landmarks'] for d in detections])[...,:2];centres=p.mean(axis=1)
                    edge=np.min(np.concatenate([p,1-p],axis=-1))
                    raw.append([1.,n,*centres.mean(axis=0),float(centres[:,0].std()),
                        float(np.prod(np.ptp(p,axis=1),axis=1).mean()),float(edge),
                        float(np.mean([d['handedness_score'] for d in detections])),0.,0.])
                else:raw.append([1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
            raw=np.array(raw)
            # Head-camera detections are assigned to anatomical hands by the
            # *available* world trajectory projected at the video time. This is
            # automatic cross-modal consistency, never independent ground truth.
            if view.startswith('head_'):
                from scipy.optimize import linear_sum_assignment
                rt,rp,_=rec.hands();ri=np.searchsorted(rt,vt,side='right')-1
                known_pose=ri>=0;ri=np.maximum(ri,0)
                projected=geometry_features(rec,ri,rp.mean(axis=2))
                extra=np.zeros((len(rows),8))
                for k,row in enumerate(rows):
                    if not known_pose[k] or not row['detections']:continue
                    expected=np.array([[projected[f'geometry_{view}_{side}_x'][k],
                                        projected[f'geometry_{view}_{side}_y'][k]] for side in ['left','right']])
                    detected=np.array([np.array(d['landmarks'])[:,:2].mean(axis=0) for d in row['detections']])
                    distance=np.linalg.norm(expected[:,None,:]-detected[None,:,:],axis=2)
                    a,b=linear_sum_assignment(distance)
                    for h,d in zip(a,b):
                        extra[k,h*4:h*4+4]=[1.,*detected[d],min(distance[h,d],2.)]
            present=(raw[1:,1]>0)&(raw[:-1,1]>0)
            raw[1:,9]=np.where(present,np.linalg.norm(np.diff(raw[:,2:4],axis=0),axis=1)/np.maximum(np.diff(vt),1e-9),0.)
            ix=np.searchsorted(vt,times,side='right')-1;known=ix>=0
            age=times-vt[np.maximum(ix,0)];known &= age<=.25
            values[known]=raw[ix[known]];values[known,8]=age[known]
            if view.startswith('head_'):
                aligned=np.zeros((len(times),8));aligned[known]=extra[ix[known]]
                for h,side in enumerate(['left','right']):
                    for j,name in enumerate(['assigned','x','y','projection_distance']):
                        result[f'video_{view}_{side}_{name}']=aligned[:,h*4+j]
        for j,name in enumerate(names):result[f'video_{view}_{name}']=values[:,j]
    return result


def geometry_features(rec,origins,centres):
    # Camera convention independently checked on training records by audit script.
    from scipy.spatial.transform import Rotation
    t=rec.streams['left']['time'];result={}
    for view in ['head_left','head_right']:
        camera=rec.streams[view+'_camera'];ix=np.searchsorted(camera['time'],t[origins],side='right')-1
        if (ix<0).any():raise ValueError('Camera pose unavailable at decision time')
        position=camera['position'][ix].reshape(-1,3);q=camera['quaternion'][ix].reshape(-1,4)
        r=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        local=np.einsum('nhi,nij->nhj',centres[origins]-position[:,None,:],r)
        z=local[...,2]
        if rec.dataset=='EgoDemo':
            meta=pd.read_parquet(rec.folder/'meta/episodes/chunk-000/file-000.parquet')
            k=np.tile(np.array(meta['camera_intrinsics/'+view].iloc[0])[:4],(len(origins),1))
            # Supplied principal point is image center; independently cross-check
            # dimensions against video manifest in geometry audit.
            width,height=2*k[:,2],2*k[:,3]
        else:
            f=pd.read_csv(rec.folder/'tables'/f'sensor_camera_{view}_intrinsic.csv')
            kix=np.searchsorted(f._offset_from_first_log_s.to_numpy(),t[origins],side='right')-1
            if (kix<0).any():raise ValueError('Intrinsics unavailable at decision time')
            f=f.iloc[kix];k=f[['K[0]','K[4]','K[2]','K[5]']].to_numpy();width=f.width.to_numpy();height=f.height.to_numpy()
        safe=np.where(abs(z)>1e-5,z,1e-5)
        x=(local[...,0]/safe*k[:,0,None]+k[:,2,None])/width[:,None]
        y=(local[...,1]/safe*k[:,1,None]+k[:,3,None])/height[:,None]
        for h,side in enumerate(['left','right']):
            result[f'geometry_{view}_{side}_x']=np.clip(x[:,h],-2,3)
            result[f'geometry_{view}_{side}_y']=np.clip(y[:,h],-2,3)
            result[f'geometry_{view}_{side}_edge']=np.clip(np.minimum.reduce([x[:,h],y[:,h],1-x[:,h],1-y[:,h]]),-2,1)
            result[f'geometry_{view}_{side}_depth']=np.clip(z[:,h],-3,3)
    return result


def dataset_frame(records,horizon,config,video_root=None):
    frames=[]
    for rec in records:
        t,p,bad=rec.hands();c=p.mean(axis=2)
        i,j=matched_targets(t,horizon,config['warmup_s'],config['target_tolerance_steps'])
        vectors,speed,turn,recent=history_features(t,c,bad,i,config['lags_s'])
        age=status_history(t,bad)
        data=dict(episode=np.repeat(rec.episode,len(i)),environment=np.repeat(rec.environment,len(i)),
            operator=np.repeat(rec.operator,len(i)),dataset=np.repeat(rec.dataset,len(i)),split=np.repeat(rec.split,len(i)),
            origin=i,target=j,time_s=t[i],target_time_s=t[j],horizon=np.repeat(horizon,len(i)),
            current_state=bad[i,0].astype(int)+2*bad[i,1].astype(int),
            target_state=bad[j,0].astype(int)+2*bad[j,1].astype(int))
        state=bad.any(axis=1)
        for name,events in [('onset',np.flatnonzero(~state[:-1]&state[1:])+1),
                            ('recovery',np.flatnonzero(state[:-1]&~state[1:])+1)]:
            count=np.r_[0,np.cumsum(np.isin(np.arange(len(t)),events))]
            data[name+'_within_horizon']=(count[j+1]-count[i+1])>0
        for ds in DATASETS:data['dataset_'+ds]=np.repeat(float(ds==rec.dataset),len(i))
        for hand,side in enumerate(['left','right']):
            data['flag_'+side]=bad[i,hand].astype(float)
            data['duration_'+side]=np.log1p(age[i,hand])
            data['motion_'+side+'_speed']=np.log1p(speed[:,hand])
            data['motion_'+side+'_turn']=np.log1p(turn[:,hand])
            data['motion_'+side+'_radius']=np.linalg.norm(p[i,hand]-c[i,hand,None,:],axis=-1).mean(axis=1)
            for window in [.1,.3,1.]:
                start=np.searchsorted(t,t[i]-window,side='left')
                count=np.r_[0,np.cumsum(bad[:,hand].astype(int))]
                data[f'history_{side}_{window}']=(count[i+1]-count[start])/(i-start+1)
        data['motion_interhand_distance']=np.linalg.norm(c[i,0]-c[i,1],axis=-1)
        velocity=vectors[...,0]
        data['motion_bimanual_cosine']=np.sum(velocity[:,0]*velocity[:,1],axis=-1)/(speed[:,0]*speed[:,1]+1e-8)
        head=rec.streams['head'];hi=np.searchsorted(head['time'],t[i],side='right')-1
        if (hi<0).any():raise ValueError('Head pose unavailable at decision time')
        hp=head['position'][hi].reshape(-1,3)
        for h,side in enumerate(['left','right']):data['motion_'+side+'_head_distance']=np.linalg.norm(c[i,h]-hp,axis=-1)
        data.update(geometry_features(rec,i,c))
        if video_root:data.update(visual_features(rec,t[i],video_root))
        frames.append(pd.DataFrame(data))
    return pd.concat(frames,ignore_index=True)


def columns_for(frame,variant):
    groups=['dataset_']
    if variant!='motion_no_flags':groups+=['flag_','duration_']
    if variant not in ['duration']:groups+=['motion_']
    if variant not in ['duration','motion_no_flags']:groups+=['history_']
    if variant in ['geometry','head_video','all_video']:groups+=['geometry_']
    if variant=='head_video':groups+=['video_head_']
    if variant=='all_video':groups+=['video_']
    return [name for name in frame if any(name.startswith(prefix) for prefix in groups)]


def weights(frame):return balanced_weights(frame.environment,frame.episode)


def fit_classifier(frame,columns,kind):
    x=frame[columns].to_numpy();y=frame.target_state.to_numpy();w=weights(frame)*len(frame)
    if kind=='logistic':
        model=Pipeline([('scale',WeightedScale()),('classifier',LogisticRegression(C=.1,max_iter=600,random_state=20261120))])
        model.fit(x,y,scale__sample_weight=w,classifier__sample_weight=w)
    else:
        model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=80,
            l2_regularization=10.,learning_rate=.06,early_stopping=False,random_state=20261120)
        model.fit(x,y,sample_weight=w)
    return model


def predict_classifier(model,frame,columns):
    raw=model.predict_proba(frame[columns].to_numpy());out=np.full((len(frame),4),1e-8)
    out[:,np.asarray(model.classes_,int)]=raw
    return out/out.sum(axis=1,keepdims=True)


def fit_baseline(train,transition=True):
    w=weights(train);y=train.target_state.to_numpy();result={}
    global_counts=np.bincount(y,weights=w,minlength=4)+1e-6;global_counts/=global_counts.sum()
    for ds in DATASETS:
        for state in (range(4) if transition else [-1]):
            use=(train.dataset==ds).to_numpy()
            if transition:use &= (train.current_state==state).to_numpy()
            # A small fixed probability-mass prior avoids exact 0/1 estimates.
            counts=np.bincount(y[use],weights=w[use],minlength=4)+global_counts/len(train)
            result[f'{ds}:{state}']=(counts/counts.sum()).tolist()
    return dict(transition=transition,probabilities=result)


def predict_baseline(model,frame):
    return np.array([model['probabilities'][f'{d}:{s if model["transition"] else -1}']
                     for d,s in zip(frame.dataset,frame.current_state)])


def temperature(probability,value):return softmax(np.log(np.clip(probability,1e-9,1))/value,axis=1)


def fit_temperature(oof,frame):
    y=(frame.target_state.to_numpy()>0).astype(float);w=weights(frame)
    objective=lambda temp:float(np.dot(w,(1-temperature(oof,temp)[:,0]-y)**2))
    result=minimize_scalar(objective,bounds=(.4,4.),method='bounded')
    return float(result.x)


def grouped_oof(frame,columns,kind):
    result=np.empty((len(frame),4));folds=[]
    # Four group folds keep entire environments together; fixed before fitting.
    for fold,(train_ix,held_ix) in enumerate(GroupKFold(n_splits=4).split(frame,groups=frame.environment)):
        train=frame.iloc[train_ix];held=frame.iloc[held_ix]
        if set(train.environment)&set(held.environment):raise AssertionError('Environment leak')
        model=fit_classifier(train,columns,kind);result[held_ix]=predict_classifier(model,held,columns)
        folds.append(dict(fold=fold,training_environments=sorted(train.environment.unique()),
                          held_environments=sorted(held.environment.unique())))
    return result,folds


def metrics(frame,probability,method):
    rows=[]
    targets={'any':(frame.target_state.to_numpy()>0).astype(int),
             'left':(frame.target_state.to_numpy()%2).astype(int),'right':(frame.target_state.to_numpy()//2).astype(int)}
    probabilities={'any':1-probability[:,0],'left':probability[:,1]+probability[:,3],
                   'right':probability[:,2]+probability[:,3]}
    for ds in [*DATASETS,'ALL']:
        domain=np.ones(len(frame),bool) if ds=='ALL' else (frame.dataset==ds).to_numpy()
        for condition in ['all','currently_good','currently_bad']:
            take=domain.copy()
            if condition!='all':take &= (frame.current_state.to_numpy()==0) if condition=='currently_good' else (frame.current_state.to_numpy()>0)
            if not take.any():continue
            for target in targets:
                y=targets[target][take];p=probabilities[target][take];w=weights(frame.loc[take])
                clipped=np.clip(p,1e-7,1-1e-7)
                rows.append(dict(dataset=ds,condition=condition,target=target,method=method,
                    n=int(take.sum()),positive=int(y.sum()),environments=frame.loc[take].environment.nunique(),
                    brier=float(np.dot(w,(p-y)**2)),log_loss=float(np.dot(w,-y*np.log(clipped)-(1-y)*np.log1p(-clipped))),
                    ap=float(average_precision_score(y,p,sample_weight=w)) if y.any() else None,
                    auc=float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))==2 else None))
    return rows


def select_threshold(train_oof,frame,false_alarm_level,mode='onset'):
    from .q1v2_state import weighted_quantile
    event=frame[mode+'_within_horizon'] if mode+'_within_horizon' in frame else ((frame.target_state>0) if mode=='onset' else (frame.target_state==0))
    if mode=='onset':negative=(frame.current_state==0)&~event;score=1-train_oof[:,0]
    else:negative=(frame.current_state>0)&~event;score=train_oof[:,0]
    f=frame[negative];w=weights(f)
    threshold=weighted_quantile(score[negative],w,1-false_alarm_level)
    # Strict > threshold handles tied baseline scores without inflated false alarms.
    return float(threshold)


def event_metrics(records,frame,probability,threshold,horizon,mode='onset'):
    events=[];false_times=[]
    scores=1-probability[:,0] if mode=='onset' else probability[:,0]
    for rec in records:
        take=(frame.episode==rec.episode).to_numpy();f=frame.loc[take];p=scores[take]
        t,_,bad=rec.hands();status=bad.any(axis=1);dt=np.median(np.diff(t))
        risk=(f.current_state.to_numpy()==0) if mode=='onset' else (f.current_state.to_numpy()>0)
        event= f[mode+'_within_horizon'].to_numpy() if mode+'_within_horizon' in f else ((f.target_state.to_numpy()>0) if mode=='onset' else (f.target_state.to_numpy()==0))
        negative=~event
        false_times.append(dict(environment=rec.environment,episode=rec.episode,
            false_alarm_fraction=float(np.mean((p[risk&negative]>threshold))) if np.any(risk&negative) else np.nan))
        for start,stop in contiguous_runs(status):
            index=start if mode=='onset' else stop
            if index==0 or index>=len(t):continue
            event_time=float(t[index])
            if len(f)==0 or event_time-horizon<float(f.time_s.min())-dt/2 or event_time>float(f.target_time_s.max())+dt/2:continue
            eligible=(f.time_s.to_numpy()>=event_time-horizon-dt/2)&(f.time_s.to_numpy()<event_time)&risk
            alarm=eligible&(p>threshold)
            lead=event_time-float(f.time_s.to_numpy()[alarm][0]) if alarm.any() else np.nan
            events.append(dict(dataset=rec.dataset,environment=rec.environment,episode=rec.episode,
                mode=mode,event_time=event_time,detected=bool(alarm.any()),lead_s=lead))
    e=pd.DataFrame(events);f=pd.DataFrame(false_times)
    macro=e.groupby(['environment','episode']).detected.mean().groupby('environment').mean().mean() if len(e) else np.nan
    return dict(events=len(e),detected=int(e.detected.sum()) if len(e) else 0,
        event_recall=float(e.detected.mean()) if len(e) else None,environment_recall=float(macro) if np.isfinite(macro) else None,
        median_lead=float(e.lead_s.median()) if len(e) and e.detected.any() else None,
        false_alarm_fraction=float(f.groupby('environment').false_alarm_fraction.mean().mean())),e
