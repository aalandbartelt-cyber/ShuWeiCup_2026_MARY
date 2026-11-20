"""Additional causal motion experts, with the original Q2 schema as a prefix.

No recording handle or future label is accepted. Qualified disclosed history is
used consistently with Q2; all new displacements vanish at a direct observation.
"""
import numpy as np
from .q2_features import CausalExperts, EXPERT_NAMES as BASIC_NAMES

EXTRA_LAGS=(2/30,.15,.3)
EXTRA_NAMES=[f'velocity_nearest_{h:g}' for h in EXTRA_LAGS]+['quadratic_anchor_13']+[f'quadratic_free_{n}' for n in (5,9,13)]
EXPERT_NAMES=list(BASIC_NAMES)+EXTRA_NAMES


class RichExperts(CausalExperts):
    def prepare(self,history):
        if history is self._history:return
        super().prepare(history)
        self.extra_velocity=np.zeros((2,3,len(EXTRA_LAGS)))
        self.extra_poly=np.zeros((2,4,3,3))
        for hand in range(2):
            indices=np.flatnonzero(~history.bad[:,hand]&np.isfinite(self.centres[:,hand]).all(axis=-1))
            if not len(indices):indices=np.flatnonzero(np.isfinite(self.centres[:,hand]).all(axis=-1))
            indices=indices[-64:];t=self.t[indices];c=self.centres[indices,hand]
            if len(t)<2:continue
            for k,lag in enumerate(EXTRA_LAGS):
                previous=int(np.argmin(abs(t[:-1]-(t[-1]-lag))))
                self.extra_velocity[hand,:,k]=(c[-1]-c[previous])/(t[-1]-t[previous])
            for k,size in enumerate((13,5,9,13)):
                tt=t[-size:]-t[-1];span=max(abs(tt[0]),1e-6);u=tt/span
                if len(tt)<3:continue
                if k==0:
                    design=np.stack([u,u*u],axis=1)
                    beta=np.linalg.lstsq(design,c[-size:]-c[-1],rcond=1e-6)[0]
                    self.extra_poly[hand,k,1]=beta[0]/span
                    self.extra_poly[hand,k,2]=beta[1]/span**2
                else:
                    design=np.stack([np.ones_like(u),u,u*u],axis=1)
                    beta=np.linalg.lstsq(design,c[-size:]-c[-1],rcond=1e-6)[0]
                    self.extra_poly[hand,k,0]=beta[0]
                    self.extra_poly[hand,k,1]=beta[1]/span
                    self.extra_poly[hand,k,2]=beta[2]/span**2

    def at(self,history,now):
        pack=super().at(history,now);age=pack['age']
        extra=[self.extra_velocity[:,:,k]*age[:,None] for k in range(len(EXTRA_LAGS))]
        extra.extend(self.extra_poly[:,k,0]+self.extra_poly[:,k,1]*age[:,None]+self.extra_poly[:,k,2]*age[:,None]**2 for k in range(4))
        extra=np.stack(extra,axis=-1);extra[age<=1e-9]=0.
        features=[];names=[]
        for k,name in enumerate(EXTRA_NAMES):
            features.append(np.linalg.norm(extra[:,:,k],axis=1));names.append('length_'+name)
        velocities=np.concatenate([self.velocity,self.extra_velocity],axis=2)
        speed=np.linalg.norm(velocities,axis=1)
        # Pairwise directions are scale and rigid-rotation invariant.
        for a in range(velocities.shape[-1]):
            for b in range(a+1,velocities.shape[-1]):
                features.append(np.sum(velocities[:,:,a]*velocities[:,:,b],axis=1)/(speed[:,a]*speed[:,b]+1e-12))
                names.append(f'rich_cos_{a}_{b}')
        pack['experts']=np.concatenate([pack['experts'],extra.astype(np.float32)],axis=-1)
        pack['features']=np.concatenate([pack['features'],np.stack(features,axis=1).astype(np.float32)],axis=1)
        pack['feature_names']=pack['feature_names']+names
        return pack
