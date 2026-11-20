"""Retain Q1's position/recent-velocity state under causal missing observations."""
import numpy as np


class StateTracker:
    """Wrap a point predictor without changing its scored point forecasts.

    The recent-velocity lag is the same 1/30 s as Q1. At a hidden past time,
    use the archived causal forecast; if its qualified numerical sample has
    since arrived, use that received sample. This is an estimated velocity,
    not an independently measured physical velocity.
    """
    def __init__(self,predictor,lag=1/30):
        if lag<=0:raise ValueError('Positive velocity lag required')
        self.predictor=predictor;self.lag=lag
        self.times=[];self.positions=[];self.states=[]

    def __call__(self,history,now):
        if self.times and now<=self.times[-1]:
            raise ValueError('State observations must advance in time')
        position=np.asarray(self.predictor(history,now),float)
        times=np.union1d(history.time[history.time<now-1e-9],self.times)
        if not len(times):raise ValueError('No past support for recent velocity')
        previous=float(times[np.argmin(np.abs(times-(now-self.lag)))])
        source=np.searchsorted(history.time,previous)
        archived=np.searchsorted(self.times,previous)
        old=np.zeros((2,3));known=[];support_time=np.full(2,previous)
        for hand in range(2):
            received=source<len(history.time) and history.time[source]==previous
            if received and not history.bad[source,hand]:
                old[hand]=history.centres[source,hand];known.append('received')
            elif archived<len(self.times) and self.times[archived]==previous:
                old[hand]=self.positions[archived][hand];known.append('archived_forecast')
            else:
                valid=(history.time<=previous)&~history.bad[:,hand]&np.isfinite(history.centres[:,hand]).all(axis=-1)
                indices=np.flatnonzero(valid)
                label='earlier_received_fallback'
                if not len(indices):
                    indices=np.flatnonzero((history.time<=previous)&np.isfinite(history.centres[:,hand]).all(axis=-1))
                    label='flagged_received_fallback'
                if not len(indices):raise ValueError('No finite past state or forecast for velocity')
                index=indices[-1];support_time[hand]=history.time[index]
                old[hand]=history.centres[index,hand];known.append(label)
        velocity=(position-old)/(now-support_time)[:,None]
        # [c_L,v_L,c_R,v_R], exactly the Q1 coordinate order.
        state=np.concatenate([np.r_[position[h],velocity[h]] for h in range(2)])
        self.times.append(float(now));self.positions.append(position.copy())
        self.states.append(dict(time_s=float(now),lag_source_s=previous,state=state,
                                velocity_support=tuple(known),velocity_support_time_s=support_time.copy()))
        return position
