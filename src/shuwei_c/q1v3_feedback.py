"""Delayed feedback radius adaptation; issued radii are never recomputed retroactively."""
import numpy as np


def adapt_radii(frame,base,levels,rate,log_bound=np.log(3.)):
    base=np.asarray(base);levels=np.asarray(levels);issued=np.zeros_like(base);multipliers=np.zeros_like(base)
    for _,g in frame.groupby(['episode','hand'],sort=False):
        g=g.sort_values('origin_s');ix=g.index.to_numpy();origin=g.origin_s.to_numpy();target=g.target_s.to_numpy();error=g.gate_normalized_error.to_numpy()
        order=np.argsort(target);pointer=0;logscale=np.zeros(len(levels));last=None
        for local,row in enumerate(ix):
            while pointer<len(order) and target[order[pointer]]<=origin[local]+1e-9:
                previous=order[pointer];assert previous<local,'A residual must belong to an already issued forecast'
                # The first update uses the declared reference unit, never a
                # sampling interval estimated from this record's future rows.
                elapsed=1/30 if last is None else max(target[previous]-last,0.)
                gain=rate*min(elapsed/(1/30),3.)
                missed=error[previous]>issued[ix[previous]]
                logscale=np.clip(logscale+gain*(missed-(1-levels)),-log_bound,log_bound)
                last=target[previous];pointer+=1
            multipliers[row]=np.exp(logscale)
            issued[row]=np.maximum.accumulate(base[row]*multipliers[row])
    return issued,multipliers
