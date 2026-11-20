"""Coarse empirical radii that separate observation age from sparse velocity history."""
from pathlib import Path
import json
import numpy as np
from .q2_calibration import hierarchical_weights,weighted_quantile


class HistoryBinnedRadius:
    """Age/gap/motion bins with explicit whole-environment support and backoff.

    This is a finite-data quantile model, not a coverage guarantee. Motion splits
    and quantiles are fitted only on the caller's training partition.
    """
    def __init__(self,levels=(.5,.8,.9,.95),rate=False,motion=False,min_environments=3):
        self.levels=np.asarray(levels,float);self.rate=bool(rate);self.motion=bool(motion)
        self.min_environments=min_environments
        self.age_edges=np.array([.05,.1,.2,.5,1.,2.])
        self.gap_edges=np.array([.05,.15,.4])

    def _bins(self,age,gap,speed):
        a=np.searchsorted(self.age_edges,age,side='left')
        g=np.searchsorted(self.gap_edges,gap,side='left')
        v=(speed>self.speed_threshold).astype(int) if self.motion else np.zeros(len(a),int)
        return a,g,v

    def fit(self,age,gap,speed,error,meta):
        age,gap,speed,error=[np.asarray(x,float) for x in (age,gap,speed,error)]
        if not all(x.shape==age.shape and np.isfinite(x).all() and (x>=0).all() for x in (gap,speed,error)):
            raise ValueError('Nonnegative finite aligned inputs required')
        positive=age>1e-9
        self.speed_threshold=weighted_quantile(speed[positive],.5,hierarchical_weights(meta.loc[positive]))
        target=error/np.maximum(age,1/30) if self.rate else error
        a,g,v=self._bins(age,gap,speed);self.cells={};self.support=[]
        groups=[('all',positive)]
        for av in np.unique(a[positive]):
            groups.append((f'a{av}',positive&(a==av)))
            for gv in np.unique(g[positive&(a==av)]):
                groups.append((f'a{av}g{gv}',positive&(a==av)&(g==gv)))
                if self.motion:
                    for vv in (0,1):groups.append((f'a{av}g{gv}v{vv}',positive&(a==av)&(g==gv)&(v==vv)))
        for key,mask in groups:
            environments=meta.loc[mask,'environment'].nunique()
            if environments<self.min_environments and key!='all':continue
            w=hierarchical_weights(meta.loc[mask])
            self.cells[key]=np.array([weighted_quantile(target[mask],q,w) for q in self.levels])
            self.support.append(dict(key=key,rows=int(mask.sum()),environments=int(environments)))
        self.fit_environments=sorted(meta.environment.unique())
        return self

    def predict(self,age,gap,speed):
        age,gap,speed=[np.asarray(x,float) for x in (age,gap,speed)]
        a,g,v=self._bins(age,gap,speed);result=np.tile(self.cells['all'],(len(age),1))
        # Broad to specific assignment implements the documented backoff.
        for av in np.unique(a):
            if f'a{av}' in self.cells:result[a==av]=self.cells[f'a{av}']
            for gv in np.unique(g[a==av]):
                mask=(a==av)&(g==gv);key=f'a{av}g{gv}'
                if key in self.cells:result[mask]=self.cells[key]
                if self.motion:
                    for vv in (0,1):
                        key=f'a{av}g{gv}v{vv}'
                        if key in self.cells:result[mask&(v==vv)]=self.cells[key]
        if self.rate:result*=np.maximum(age,1/30)[:,None]
        result[age<=1e-9]=0
        return np.maximum.accumulate(result,axis=1)

    def save(self,path):
        Path(path).write_text(json.dumps(dict(levels=self.levels.tolist(),rate=self.rate,motion=self.motion,
            age_edges=self.age_edges.tolist(),gap_edges=self.gap_edges.tolist(),speed_threshold=self.speed_threshold,
            min_environments=self.min_environments,fit_environments=self.fit_environments,
            cells={k:v.tolist() for k,v in self.cells.items()},support=self.support),indent=2)+'\n')

    @classmethod
    def load(cls,path):
        record=json.loads(Path(path).read_text())
        model=cls(record['levels'],record['rate'],record['motion'],record['min_environments'])
        for name in ('age_edges','gap_edges'):
            setattr(model,name,np.asarray(record[name],float))
        for name in ('speed_threshold','fit_environments','support'):
            setattr(model,name,record[name])
        model.cells={key:np.asarray(value,float) for key,value in record['cells'].items()}
        return model


def scale_path_radius(radius,factors,age):
    """Match the floor in path scores, while preserving known-current zero error."""
    result=np.maximum(np.asarray(radius,float),1e-5)*np.asarray(factors,float)[None,:]
    result[np.asarray(age)<=1e-9]=0.
    return np.maximum.accumulate(result,axis=1)
