"""Q1 v2: matched physical-time forecasts and train-group calibrated sets.

AI-assisted by Codex, 2026-11-20. Conditional numerical references are not
independent physical truth. All predictor features are prefix measurable.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from .data import balanced_weights
from .motion import estimate_noise, filter_record


def nearest_indices(time, queries):
    """Nearest existing sample; deterministic earlier-sample tie break."""
    t = np.asarray(time); q = np.asarray(queries)
    right = np.clip(np.searchsorted(t, q), 0, len(t)-1)
    left = np.maximum(right-1, 0)
    return np.where(abs(t[left]-q) <= abs(t[right]-q), left, right)


def matched_targets(time, horizon, warmup=.4, tolerance_steps=.55):
    t = np.asarray(time)
    origins = np.flatnonzero((t >= t[0]+warmup-1e-9) & (np.arange(len(t)) >= 2))
    desired = t[origins]+horizon
    targets = nearest_indices(t, desired)
    valid = ((targets > origins) & (desired <= t[-1]+1e-9) &
             (abs(t[targets]-desired) <= tolerance_steps*np.median(np.diff(t))+1e-9))
    return origins[valid], targets[valid]


def history_features(time, centres, bad, origins, lags):
    """Vector velocity bases plus scalar scale covariates, without future reads."""
    t, c, i = np.asarray(time), np.asarray(centres), np.asarray(origins)
    vectors=[]; history_bad=np.asarray(bad)[i].copy()
    for lag in lags:
        previous=np.minimum(nearest_indices(t, t[i]-lag), i-1)
        previous=np.maximum(previous, 0)
        dt=np.maximum(t[i]-t[previous], 1e-9)
        vectors.append((c[i]-c[previous])/dt[:,None,None])
        history_bad |= bad[previous]
    x=np.stack(vectors,axis=-1)
    speed=np.linalg.norm(x[...,0],axis=-1)
    turn=np.linalg.norm(x[...,0]-x[...,1],axis=-1)
    return x, speed, turn, history_bad


def frame_for(rec, horizon, config):
    t,p,bad=rec.hands();c=p.mean(axis=2)
    i,j=matched_targets(t,horizon,config['warmup_s'],config['target_tolerance_steps'])
    x,speed,turn,history_bad=history_features(t,c,bad,i,config['lags_s'])
    return dict(t=t,c=c,bad=bad,i=i,j=j,x=x,speed=speed,turn=turn,
                history_bad=history_bad,dt=t[j]-t[i])


def linear_moments(records,horizon,config,noise):
    rows=[]
    for rec in records:
        f=frame_for(rec,horizon,config)
        valid=~f['bad'][f['j']] & ~f['history_bad']
        x=f['x'][valid]/noise[rec.dataset]['hand_scale']
        y=((f['c'][f['j']]-f['c'][f['i']])/f['dt'][:,None,None])[valid]/noise[rec.dataset]['hand_scale']
        x=x.reshape(-1,len(config['lags_s']));y=y.ravel()
        if not len(y):continue
        rows.append(dict(dataset=rec.dataset,environment=rec.environment,episode=rec.episode,
                         gram=x.T@x/len(y),target=x.T@y/len(y)))
    return rows


def fit_linear(moments,method,penalty=0.):
    w=balanced_weights([r['environment'] for r in moments],[r['episode'] for r in moments])
    g=np.einsum('n,nij->ij',w,np.array([r['gram'] for r in moments]))
    b=np.einsum('n,ni->i',w,np.array([r['target'] for r in moments]))
    if method=='scalar_damped':return np.array([b[0]/max(g[0,0],1e-12),0.,0.,0.])
    prior=np.array([1.,0.,0.,0.])
    return np.linalg.solve(g+penalty*np.eye(4),b+penalty*prior)


def one_euro(time,centres,mincutoff,beta):
    """Speed-adaptive smoothing (Casiez et al.); output velocity is causal.

    Norm-based cutoff is shared across xyz to preserve rotation equivariance.
    This is a vector adaptation of 1 euro, not a claim of identical scalar code.
    """
    t,c=np.asarray(time),np.asarray(centres)
    out=np.empty_like(c);velocity=np.zeros_like(c);out[0]=c[0];derivative=np.zeros_like(c[0])
    for i in range(1,len(t)):
        dt=t[i]-t[i-1];a=1/(1+1/(2*np.pi*dt))
        derivative=a*(c[i]-c[i-1])/dt+(1-a)*derivative
        cutoff=mincutoff+beta*np.linalg.norm(derivative,axis=-1)
        smoothing=(1/(1+1/(2*np.pi*cutoff*dt)))[:,None]
        out[i]=smoothing*c[i]+(1-smoothing)*out[i-1]
        velocity[i]=(out[i]-out[i-1])/dt
    return out,velocity


def prediction_for(rec,f,spec,noise,cache=None):
    method=spec['method'];i=f['i'];dt=f['dt']
    if method=='constant':return f['c'][i].copy()
    if method=='linear':return f['c'][i]+f['x'][...,0]*dt[:,None,None]
    if method in ['scalar_damped','multiscale']:
        return f['c'][i]+np.einsum('nhck,k->nhc',f['x'],np.array(spec['coefficients']))*dt[:,None,None]
    key=(rec.episode,json.dumps(spec,sort_keys=True))
    if cache is not None and key in cache:position,velocity=cache[key]
    elif method=='quality_filter':
        dsnoise=dict(noise[rec.dataset]);dsnoise['acceleration']*=spec['q_scale'];dsnoise['r']*=spec['r_scale']
        dsnoise['bad_multiplier']=spec['bad_multiplier']
        state,_=filter_record(rec,dsnoise,quality=True);position,velocity=state[:,:,0],state[:,:,1]
        if cache is not None:cache[key]=(position,velocity)
    elif method=='one_euro':
        position,velocity=one_euro(f['t'],f['c'],spec['mincutoff'],spec['beta'])
        if cache is not None:cache[key]=(position,velocity)
    else:raise ValueError(method)
    return position[i]+velocity[i]*dt[:,None,None]


def error_rows(rec,f,prediction,method,horizon,scale):
    error=np.linalg.norm(prediction-f['c'][f['j']],axis=-1)
    rows=[]
    for hand in range(2):
        valid=~f['bad'][f['j'],hand]&np.isfinite(error[:,hand])
        for k in np.flatnonzero(valid):
            rows.append(dict(dataset=rec.dataset,environment=rec.environment,episode=rec.episode,
                split=rec.split,hand=hand,method=method,horizon=horizon,origin=int(f['i'][k]),
                target=int(f['j'][k]),origin_s=float(f['t'][f['i'][k]]),target_s=float(f['t'][f['j'][k]]),
                actual_horizon=float(f['dt'][k]),error=float(error[k,hand]),scale=scale,
                speed=float(f['speed'][k,hand]),turn=float(f['turn'][k,hand]),
                recent_bad=bool(f['history_bad'][k,hand])))
    return rows


def grouped_summary(frame):
    rows=[]
    for (method,horizon,dataset),g in frame.groupby(['method','horizon','dataset']):
        for context in ['all','recent_bad','clean_history']:
            f=g if context=='all' else g[g.recent_bad == (context=='recent_bad')]
            if not len(f):continue
            hand=f.groupby(['environment','episode','hand']).agg(mean_error=('error','mean'),p90=('error',lambda x:np.quantile(x,.9)))
            env=hand.groupby(['environment','episode']).mean().groupby('environment').mean()
            rows.append(dict(method=method,horizon=horizon,dataset=dataset,context=context,
                mean_error=float(env.mean_error.mean()),mean_record_p90=float(env.p90.mean()),
                environments=len(env),episodes=f.episode.nunique(),hand_predictions=len(f)))
    return pd.DataFrame(rows)


def selection_score(frame):
    # Same equal-hand / episode / environment aggregation as the reported error.
    f=frame.assign(normalized=frame.error/frame.scale)
    return float(f.groupby(['environment','episode','hand']).normalized.mean().groupby(
        ['environment','episode']).mean().groupby('environment').mean().mean())


def scale_inputs(frame):
    # Nonnegative regression separates baseline scale, motion and recent bad flags.
    distance=frame.speed.to_numpy()*frame.actual_horizon.to_numpy()/frame.scale.to_numpy()
    turning=frame.turn.to_numpy()*frame.actual_horizon.to_numpy()/frame.scale.to_numpy()
    return np.column_stack([np.ones(len(frame)),distance,turning,frame.recent_bad.to_numpy(float)])


def fit_scale(frame):
    x=scale_inputs(frame);y=(frame.error/frame.scale).to_numpy()
    w=balanced_weights(frame.environment,frame.episode)
    # Tiny ridge is numeric regularization, not validation-tuned.
    a=np.vstack([x*np.sqrt(w[:,None]),np.eye(4)*1e-4]);b=np.r_[y*np.sqrt(w),np.zeros(4)]
    return nnls(a,b)[0]


def weighted_quantile(values,weights,level):
    order=np.argsort(values);v=np.asarray(values)[order];w=np.asarray(weights)[order]
    return float(v[min(np.searchsorted(np.cumsum(w)/w.sum(),level),len(v)-1)])


def calibrate(oof,levels):
    """Leave-environment scale estimation followed by weighted residual quantiles.

    Uses OOF predictions from fixed validation-selected hyperparameters. Quantiles
    are empirical, not an exchangeability-based finite-sample guarantee.
    """
    ratio=np.empty(len(oof));x=scale_inputs(oof)
    for env in oof.environment.unique():
        held=(oof.environment==env).to_numpy();coef=fit_scale(oof.loc[~held])
        ratio[held]=(oof.error/oof.scale).to_numpy()[held]/np.maximum(x[held]@coef,1e-4)
    weights=balanced_weights(oof.environment,oof.episode)
    return dict(coefficients=fit_scale(oof).tolist(),levels=levels,
        adaptive_quantiles=[weighted_quantile(ratio,weights,level) for level in levels],
        fixed_quantiles=[weighted_quantile((oof.error/oof.scale).to_numpy(),weights,level) for level in levels],
        calibration_environments=int(oof.environment.nunique()),
        interpretation='empirical leave-environment calibration; imperfect reference; no independent physical guarantee')


def evaluate_sets(frame,calibration):
    x=scale_inputs(frame);adaptive=np.maximum(x@np.array(calibration['coefficients']),1e-4)*frame.scale.to_numpy()
    rows=[]
    for scheme in ['fixed','adaptive']:
        base=frame.scale.to_numpy() if scheme=='fixed' else adaptive
        for level,q in zip(calibration['levels'],calibration[scheme+'_quantiles']):
            radius=base*q;error=frame.error.to_numpy()
            values=frame[['dataset','environment','episode','hand','recent_bad']].copy()
            values['coverage']=error<=radius;values['radius']=radius
            values['pinball']=np.where(error>=radius,level*(error-radius),(1-level)*(radius-error))
            for dataset in [*frame.dataset.unique(),'ALL']:
                sub=values if dataset=='ALL' else values[values.dataset==dataset]
                for context in ['all','recent_bad','clean_history']:
                    g=sub if context=='all' else sub[sub.recent_bad == (context=='recent_bad')]
                    if not len(g):continue
                    v=g.groupby(['environment','episode','hand'])[['coverage','radius','pinball']].mean().groupby(
                        ['environment','episode']).mean().groupby('environment').mean().mean()
                    rows.append(dict(dataset=dataset,context=context,scheme=scheme,nominal=level,
                        coverage=float(v.coverage),radius=float(v.radius),pinball=float(v.pinball),
                        environments=g.environment.nunique(),n=len(g)))
    return rows


class PrefixPredictor:
    """Q2-facing interface: receives only available hand observations."""
    def __init__(self,models,config):self.models=models;self.config=config

    def forecast(self,time,centres,bad,query_time,dataset,level=.9):
        t=np.asarray(time);c=np.asarray(centres);b=np.asarray(bad)
        if len(t)<2 or np.any(np.diff(t)<=0):raise ValueError('Need ordered available observations')
        if query_time<t[-1]-1e-9:raise ValueError('Forecast cannot precede latest received observation')
        age=max(float(query_time-t[-1]),0.)
        # Coefficients/calibration are interpolated at declared horizon knots.
        knots=np.array(self.config['horizons_s']);upper=min(np.searchsorted(knots,age),len(knots)-1)
        lower=max(upper-1,0);fraction=0. if upper==lower else np.clip((age-knots[lower])/(knots[upper]-knots[lower]),0,1)
        predictions=[];radii=[]
        for k in [lower,upper]:
            entry=self.models['horizons'][str(float(knots[k]))];spec=entry['selected']
            x,speed,turn,history_bad=history_features(t,c,b,np.array([len(t)-1]),self.config['lags_s'])
            method=spec['method']
            if method=='constant':position=c[-1]
            elif method=='linear':position=c[-1]+age*x[0,...,0]
            elif method in ['scalar_damped','multiscale']:
                position=c[-1]+age*np.einsum('hck,k->hc',x[0],np.array(spec['coefficients']))
            elif method=='one_euro':
                smooth,v=one_euro(t,c,spec['mincutoff'],spec['beta']);position=smooth[-1]+age*v[-1]
            elif method=='quality_filter':
                from .motion import predict,update
                noise=dict(self.models['noise'][dataset]);noise['acceleration']*=spec['q_scale'];noise['r']*=spec['r_scale'];noise['bad_multiplier']=spec['bad_multiplier']
                state=np.zeros((2,2,3));state[:,0]=c[0]
                cov=np.tile(np.diag([noise['r'],noise['acceleration']]),(2,1,1))
                for j in range(len(t)):
                    if j:state,cov=predict(state,cov,t[j]-t[j-1],noise)
                    state,cov=update(state,cov,c[j],b[j],noise)
                position=state[:,0]+age*state[:,1]
            else:raise ValueError(method)
            scale=self.models['noise'][dataset]['hand_scale']
            cal=entry['calibration'];covariates=np.column_stack([np.ones(2),speed[0]*age/scale,turn[0]*age/scale,history_bad[0]])
            q=np.interp(level,cal['levels'],cal[entry['set_scheme']+'_quantiles'])
            radius=(np.maximum(covariates@np.array(cal['coefficients']),1e-4)*scale if entry['set_scheme']=='adaptive' else np.full(2,scale))*q
            predictions.append(position);radii.append(radius)
        return dict(position=(1-fraction)*predictions[0]+fraction*predictions[1],
                    radius=(1-fraction)*radii[0]+fraction*radii[1],age=age,
                    extrapolation=age>knots[-1],nominal=level)
