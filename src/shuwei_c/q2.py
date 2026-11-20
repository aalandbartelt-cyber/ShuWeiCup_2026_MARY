"""Artificial missing-data replay with delayed historical measurements.

AI: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
All modalities hidden protocol here; video-available protocol needs visual features.
"""
import numpy as np
from .motion import predict,update,filter_record


def predict_received(initial_state,initial_cov,origin,now,received,noise,method,history):
    if method in ['constant','linear']:
        available=[*history,*[(t,y) for t,y,b in received]]
        available=sorted(available,key=lambda row:row[0])
        t,y=available[-1]
        velocity=np.zeros_like(y)
        if method=='linear' and len(available)>1:
            t0,y0=available[-2];velocity=(y-y0)/max(t-t0,1e-6)
        state=np.stack([y+velocity*(now-t),velocity],axis=1)
        _,cov=predict(initial_state,initial_cov,now-origin,noise)
        return state,cov
    state,cov=initial_state.copy(),initial_cov.copy();time=origin
    for stamp,y,bad in sorted(received,key=lambda row:row[0]):
        state,cov=predict(state,cov,stamp-time,noise)
        state,cov=update(state,cov,y,bad,noise,quality=method!='no_quality')
        time=stamp
    return predict(state,cov,now-time,noise)


def replay_window(time,centres,bad,start,stop,initial_state,initial_cov,noise,
                  cap,delay,method='filter',trigger='periodic',quantile=1.645,calibration=None):
    """Only the response queue reads masked values; trigger sees no hidden values."""
    origin=float(time[start-1]);end=float(time[stop-1])
    received=[];pending=[];requests=[];predictions=[];variances=[];half_widths=[]
    history=[(float(time[start-2]),centres[start-2]),(origin,centres[start-1])]
    periodic=set(np.linspace(start,stop-1,cap+2,dtype=int)[1:-1]) if cap else set()
    last_request=-float('inf')
    for i in range(start,stop):
        now=float(time[i])
        # Deliver sample AT request time, not a new sample AT delivery time.
        for request in pending[:]:
            if request['return_time']<=now+1e-9:
                ix=request['index'];received.append((float(time[ix]),centres[ix].copy(),bad[ix].copy()))
                pending.remove(request)
        state,cov=predict_received(initial_state,initial_cov,origin,now,received,noise,method,history)
        request=False
        if len(requests)<cap:
            if trigger=='periodic':request=i in periodic
            elif trigger=='width':
                age=now-max([origin]+[row[0] for row in received])
                width=np.interp(age,calibration['ages_s'],calibration['half_widths']) if calibration else quantile*np.sqrt(np.max(cov[:,0,0]))
                request=width>noise['hand_scale'] and now-last_request>=max(delay,.1)
            elif trigger=='information_gain':
                # Integrated expected squared-loss reduction in hand-scale units.
                remain=max(end-now-delay,0.)
                posterior=cov.copy()
                for h in range(2):
                    k=cov[h,:,0]/(cov[h,0,0]+noise['r'])
                    posterior[h]-=np.outer(k,cov[h,0,:])
                _,prior_end=predict(state,cov,remain+delay,noise)
                _,post_end=predict(state,posterior,remain+delay,noise)
                gain=float(np.sum(prior_end[:,0,0]-post_end[:,0,0]))*3*remain/noise['hand_scale']**2
                request=gain>1.0 and not pending and now-last_request>=.1
            else:raise ValueError(trigger)
        if request:
            item=dict(index=i,request_time=now,return_time=now+delay)
            requests.append(item.copy());last_request=now
            if delay<=1e-12:
                received.append((now,centres[i].copy(),bad[i].copy()))
                state,cov=predict_received(initial_state,initial_cov,origin,now,received,noise,method,history)
            else:pending.append(item)
        predictions.append(state[:,0].copy());variances.append(cov[:,0,0].copy()+noise['r'])
        age=now-max([origin]+[row[0] for row in received])
        half_widths.append(np.interp(age,calibration['ages_s'],calibration['half_widths']) if calibration else None)
    prediction=np.array(predictions);variance=np.array(variances)
    truth=centres[start:stop];valid=~bad[start:stop]&np.isfinite(truth).all(axis=-1)
    error=np.linalg.norm(prediction-truth,axis=-1)
    width=np.broadcast_to(np.array(half_widths)[:,None,None],(*variance.shape,1)) if calibration else quantile*np.sqrt(variance)[...,None]
    inside=np.all(np.abs(prediction-truth)<=width,axis=-1)
    # Central 90% interval score per coordinate; calibration factor is evaluated empirically.
    lo,hi=prediction-width,prediction+width
    interval=(hi-lo)+20*np.maximum(lo-truth,0)+20*np.maximum(truth-hi,0)
    return dict(error=float(error[valid].mean()) if valid.any() else None,
                coverage=float(inside[valid].mean()) if valid.any() else None,
                width=float((2*width[...,0])[valid].mean()) if valid.any() else None,
                interval_score=float(interval[valid].mean()) if valid.any() else None,
                valid_hand_frames=int(valid.sum()),requests=len(requests),
                responses_within_window=sum(r['return_time']<=end+1e-9 for r in requests),
                request_cost=float(len(requests)),latency_sum_s=float(len(requests)*delay),
                predictions=prediction,request_log=requests)


def evaluate_records(records,noise,calibration,config):
    rows=[]
    variants=[('constant','periodic'),('linear','periodic'),('filter','periodic'),
              ('filter','width'),('filter','information_gain'),('no_quality','periodic')]
    horizons=config['main_windows_s']+config['stress_windows_s']
    delays=[config['main_delay_s']]+config['sensitivity_delays_s']
    for rec in records:
        time,points,bad=rec.hands();centres=points.mean(axis=2)
        filtered={quality:filter_record(rec,noise[rec.dataset],quality) for quality in [True,False]}
        for start_time in np.arange(config['warmup_s'],time[-1],config['window_stride_s']):
            start=int(np.searchsorted(time,start_time))
            if start<2:continue
            for horizon in horizons:
                stop=int(np.searchsorted(time,time[start-1]+horizon,side='right'))
                if time[start-1]+horizon>time[-1] or stop<=start:continue
                for cap in config['request_caps']:
                    for delay in delays:
                        for method,trigger in variants:
                            if cap==0 and trigger!='periodic':continue
                            states,covs=filtered[method!='no_quality']
                            result=replay_window(time,centres,bad,start,stop,states[start-1],covs[start-1],noise[rec.dataset],
                                cap,delay,method,trigger,calibration=calibration[rec.dataset][method])
                            result.pop('predictions');result.pop('request_log')
                            rows.append(dict(episode=rec.episode,environment=rec.environment,dataset=rec.dataset,
                                split=rec.split,start_s=float(time[start-1]),horizon_s=horizon,cap=cap,delay_s=delay,
                                method=method,trigger=trigger,protocol='all_modalities_hidden',**result))
    return rows
