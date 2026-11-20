"""Training-only age-conditioned correction of cross-fitted sparse radii."""
import json
from pathlib import Path
import numpy as np
from .q2_calibration import hierarchical_weights, weighted_quantile


class AgeRatioCorrection:
    """Empirical residual/radius quantiles, not a simultaneous coverage theorem."""
    def __init__(self,levels=(.5,.8,.9,.95),edges=(0.,.05,.1,.2,.5,1.,2.,float('inf')),min_environments=3):
        self.levels=np.asarray(levels,float);self.edges=np.asarray(edges,float)
        self.min_environments=min_environments
        if np.any(np.diff(self.edges)<=0) or np.any(np.diff(self.levels)<=0):
            raise ValueError('Bin edges and levels must increase')

    def fit(self,age,errors,radii,meta):
        age,errors,radii=np.asarray(age),np.asarray(errors),np.asarray(radii)
        if radii.shape!=(len(age),len(self.levels)) or errors.shape!=age.shape:
            raise ValueError('Misaligned correction inputs')
        if not np.isfinite(radii).all() or np.any(radii<0) or not np.isfinite(errors).all() or np.any(errors<0):
            raise ValueError('Invalid radii/errors')
        positive=age>1e-9
        if not positive.any():
            raise ValueError('No unobserved rows for correction')
        ratio=errors[:,None]/np.maximum(radii,1e-5)
        self.global_ratio=np.array([weighted_quantile(ratio[positive,j],q,hierarchical_weights(meta.loc[positive]))
                                    for j,q in enumerate(self.levels)])
        self.ratios=[];self.counts=[]
        bins=np.searchsorted(self.edges[1:],age,side='left')
        for i in range(len(self.edges)-1):
            mask=positive&(bins==i)
            enough=meta.loc[mask,'environment'].nunique()>=self.min_environments
            values=np.array([weighted_quantile(ratio[mask,j],q,hierarchical_weights(meta.loc[mask]))
                             for j,q in enumerate(self.levels)]) if enough else self.global_ratio.copy()
            self.ratios.append(np.maximum(values,1e-6))
            self.counts.append(dict(rows=int(mask.sum()),environments=int(meta.loc[mask,'environment'].nunique()),fallback=not enough))
        self.ratios=np.asarray(self.ratios);self.fit_environments=sorted(meta.environment.unique())
        return self

    def predict(self,age,radii):
        age=np.asarray(age);radii=np.asarray(radii)
        bins=np.searchsorted(self.edges[1:],age,side='left').clip(0,len(self.ratios)-1)
        result=np.maximum.accumulate(np.maximum(radii,1e-5)*self.ratios[bins],axis=1)
        result[age<=1e-9]=0
        return result

    def save(self,path):
        Path(path).write_text(json.dumps(dict(levels=self.levels.tolist(),edges=[None if np.isinf(x) else float(x) for x in self.edges],
            ratios=self.ratios.tolist(),global_ratio=self.global_ratio.tolist(),counts=self.counts,
            fit_environments=self.fit_environments,min_environments=self.min_environments,
            interpretation='Empirical cross-fitted residual/radius adjustment; no distribution-free or joint-path guarantee'),indent=2)+'\n')
