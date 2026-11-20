"""Stateful issued-ball ledger and already-arrived residual summaries.

Callers supply references explicitly; this object has no dataset/file access.
Times must advance monotonically, including delayed reference arrival times.
"""
import numpy as np


class StreamingBallState:
    def __init__(self,scale,levels=(.5,.8,.9,.95),rate=.01,flagged_floor=None):
        if scale<=0:raise ValueError('Positive hand scale required')
        self.scale=float(scale);self.levels=np.asarray(levels,float);self.rate=rate
        self.floor=None if flagged_floor is None else np.asarray(flagged_floor,float)
        self.logscale=np.zeros((2,len(levels)));self.averages=np.full((2,3),np.nan)
        self.last=np.full(2,np.nan);self.clock=-np.inf;self.pending={};self.next_id=0

    def _advance(self,now):
        if not np.isfinite(now) or now<self.clock-1e-9:raise ValueError('Cannot query an earlier time after observing a later arrival')
        self.clock=max(self.clock,float(now))

    def features(self,now):
        self._advance(now);out={}
        for j,tau in enumerate([.3,1.,3.]):out[f'past_residual_ewm_{tau}']=np.log1p(self.averages[:,j])
        known=np.isfinite(self.last);out['past_residual_known']=known.astype(float)
        out['past_residual_age']=np.where(known,now-self.last,0.)
        return out

    def issue(self,now,query_time,position,base_radius,current_bad):
        self._advance(now)
        if query_time<=now:raise ValueError('A forecast query must be later than its issuance')
        point=np.asarray(position,float);base=np.asarray(base_radius,float);bad=np.asarray(current_bad,bool)
        if point.shape!=(2,3) or base.shape!=self.logscale.shape or bad.shape!=(2,):raise ValueError('Invalid hand/level dimensions')
        if not np.isfinite(point).all() or not np.isfinite(base).all() or (base<0).any():raise ValueError('Finite positions and nonnegative normalized radii required')
        base=np.maximum.accumulate(base,axis=1)
        if self.floor is not None:base[bad]=np.maximum(base[bad],self.floor)
        radius=np.maximum.accumulate(base*np.exp(self.logscale),axis=1)
        key=self.next_id;self.next_id+=1
        self.pending[key]=dict(query_time=float(query_time),position=point.copy(),normalized_radius=radius.copy())
        return key,radius*self.scale

    def observe(self,key,reference_time,position,qualified,arrived_at=None):
        arrival=reference_time if arrived_at is None else arrived_at
        if arrival<reference_time-1e-9:raise ValueError('A reference cannot arrive before it exists')
        self._advance(arrival);forecast=self.pending[key]
        if abs(reference_time-forecast['query_time'])>1e-8:raise ValueError('Reference time does not match this issued query')
        point=np.asarray(position,float);valid=np.asarray(qualified,bool)
        if point.shape!=(2,3) or valid.shape!=(2,):raise ValueError('Invalid reference dimensions')
        error=np.linalg.norm(point-forecast['position'],axis=1)/self.scale
        for hand in range(2):
            if not valid[hand] or not np.isfinite(error[hand]):continue
            first=not np.isfinite(self.last[hand]);elapsed=1/30 if first else max(reference_time-self.last[hand],0.)
            gain=self.rate*min(elapsed/(1/30),3.)
            missed=error[hand]>forecast['normalized_radius'][hand]
            self.logscale[hand]=np.clip(self.logscale[hand]+gain*(missed-(1-self.levels)),-np.log(3.),np.log(3.))
            if first:self.averages[hand]=error[hand]
            else:
                alpha=-np.expm1(-elapsed/np.array([.3,1.,3.]));self.averages[hand]=(1-alpha)*self.averages[hand]+alpha*error[hand]
            self.last[hand]=reference_time if first else max(reference_time,self.last[hand])
        del self.pending[key]

    def discard(self,key):
        """An unavailable/expired reference produces no update."""
        del self.pending[key]
