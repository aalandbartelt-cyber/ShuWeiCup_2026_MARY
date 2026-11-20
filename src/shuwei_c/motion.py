"""Causal centroid/velocity filter and training-only noise estimates.

AI: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
Coordinate units remain the record's native units, not asserted centimetres.
"""
import numpy as np
from scipy.stats import norm
from .data import balanced_weights


def estimate_noise(records):
    stats=[]
    for rec in records:
        t,p,bad=rec.hands();c=p.mean(axis=2);dt=np.diff(t)
        clean=~bad[2:]&~bad[1:-1]&~bad[:-2]
        accel=(np.diff(c,axis=0)[1:]/dt[1:,None,None]-np.diff(c,axis=0)[:-1]/dt[:-1,None,None])/((dt[1:]+dt[:-1])/2)[:,None,None]
        second=c[2:]-2*c[1:-1]+c[:-2]
        # Robust moment proxies; numerical self-consistency, not sensor-ground-truth noise.
        r=np.median(second[clean]**2)/6 if clean.any() else 1e-5
        a=np.median(accel[clean]**2) if clean.any() else 1.
        radius=np.linalg.norm(p-c[:,:,None,:],axis=-1)
        stats.append(dict(dataset=rec.dataset,environment=rec.environment,episode=rec.episode,
                          r=max(float(r),1e-8),acceleration=max(float(a),1e-4),
                          hand_scale=max(float(np.median(radius)),1e-4)))
    result={}
    for ds in sorted({r['dataset'] for r in stats}):
        rows=[r for r in stats if r['dataset']==ds]
        w=balanced_weights([r['environment'] for r in rows],[r['episode'] for r in rows])
        result[ds]={k:float(np.exp(np.average(np.log([r[k] for r in rows]),weights=w))) for k in ['r','acceleration','hand_scale']}
    return result


def transition(dt, acceleration):
    f=np.array([[1.,dt],[0.,1.]])
    # Piecewise random acceleration with variance estimated on training records.
    q=acceleration*np.array([[dt**4/4,dt**3/2],[dt**3/2,dt**2]])
    return f,q


def predict(state,cov,dt,noise):
    f,q=transition(max(float(dt),0.),noise['acceleration'])
    return np.einsum('ij,hjk->hik',f,state),np.einsum('ij,hjk,lk->hil',f,cov,f)+q[None,:,:]


def update(state,cov,measurement,bad,noise,quality=True):
    state,cov=state.copy(),cov.copy()
    for h in range(2):
        if not np.isfinite(measurement[h]).all():continue
        r=noise['r']*(noise.get('bad_multiplier',100) if quality and bad[h] else 1)
        k=cov[h,:,0]/(cov[h,0,0]+r)
        state[h]+=k[:,None]*(measurement[h]-state[h,0])[None,:]
        a=np.eye(2);a[:,0]-=k
        cov[h]=a@cov[h]@a.T+r*np.outer(k,k)
    return state,cov


def filter_record(record,noise,quality=True):
    t,p,bad=record.hands();c=p.mean(axis=2)
    state=np.zeros((2,2,3));state[:,0]=c[0]
    cov=np.tile(np.diag([noise['r'],noise['acceleration']]),(2,1,1))
    states,covs=[],[]
    for i in range(len(t)):
        if i:state,cov=predict(state,cov,t[i]-t[i-1],noise)
        state,cov=update(state,cov,c[i],bad[i],noise,quality)
        states.append(state.copy());covs.append(cov.copy())
    return np.array(states),np.array(covs)


def calibration(records,noise,horizons):
    """Training-only rolling-origin calibration, balanced by environment/episode.

    Noise estimates share training data: this is development calibration, not an
    independent coverage guarantee. Validation coverage is always measured.
    """
    horizons=sorted(set([0.,1/30,2/30,.1]+list(horizons)))
    methods=['constant','linear','filter','no_quality']
    pools={(ds,m,h):([],[],[]) for ds in noise for m in methods for h in horizons}
    for rec in records:
        t,p,bad=rec.hands();c=p.mean(axis=2)
        fitted={q:filter_record(rec,noise[rec.dataset],q) for q in [True,False]}
        for i in range(30,len(t),30):
            for horizon in horizons:
                j=np.searchsorted(t,t[i]+horizon)
                if j>=len(t):continue
                valid=~bad[j]&~bad[i]&~bad[i-1]
                if not valid.any():continue
                for method in methods:
                    if method=='constant':value=c[i]
                    elif method=='linear':value=c[i]+(c[i]-c[i-1])/(t[i]-t[i-1])*(t[j]-t[i])
                    else:
                        states,covs=fitted[method!='no_quality']
                        state,_=predict(states[i],covs[i],t[j]-t[i],noise[rec.dataset]);value=state[:,0]
                    errors=np.max(np.abs(value-c[j]),axis=-1)[valid]
                    values,env,ep=pools[(rec.dataset,method,horizon)]
                    values.extend(errors.tolist());env.extend([rec.environment]*len(errors));ep.extend([rec.episode]*len(errors))
    result={ds:{m:dict(ages_s=horizons,half_widths=[]) for m in methods} for ds in noise}
    for (ds,method,horizon),(values,env,ep) in pools.items():
        order=np.argsort(values);w=balanced_weights(env,ep)[order]
        q=np.asarray(values)[order][min(np.searchsorted(np.cumsum(w),.9),len(w)-1)]
        result[ds][method]['half_widths'].append(max(float(q),1e-8))
    return result
