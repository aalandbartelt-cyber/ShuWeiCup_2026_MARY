"""CPU deployment adapters for frozen numerical point-predictor candidates."""
from pathlib import Path
import numpy as np
import torch
from .q2_features import CausalExperts, transform_features


def _silu(value):
    return value/(1+np.exp(-np.clip(value,-80,80)))


class SparseGate:
    """A small gate evaluated with NumPy; no recordings or reference handles."""
    def __init__(self, models, dataset):
        self.models=[];self.dataset=dataset;self.extractor=CausalExperts(dataset)
        for source in models:
            model=torch.load(source,map_location='cpu',weights_only=False) if isinstance(source,(str,Path)) else source
            kind=model['kind'];prefix='gate' if kind=='static_gate' else 'net'
            if kind not in ('static_gate','mixture','residual'):
                raise ValueError('Unsupported deployment gate')
            state={k:v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
                   for k,v in model['state_dict'].items()}
            layers=[(state[f'{prefix}.{j}.weight'],state[f'{prefix}.{j}.bias'])
                    for j in (0,2,4) if f'{prefix}.{j}.weight' in state]
            self.models.append(dict(kind=kind,layers=layers,names=model['feature_names'],
                                    mean=model.get('feature_mean',model.get('mean')),
                                    std=model.get('feature_std',model.get('std')),
                                    scale=float(model['noise'][dataset]['hand_scale'])))
        if not self.models:
            raise ValueError('At least one fitted checkpoint is required')

    def displacement(self, pack):
        results=[]
        for model in self.models:
            if list(pack['feature_names'])!=list(model['names']):
                raise ValueError('Feature schema differs from trained checkpoint')
            scale=np.float32(model['scale'])
            x=transform_features(pack['features'],pack['feature_names'],scale)
            x=np.clip((x-model['mean'])/model['std'],-10,10).astype(np.float32)
            for i,(weight,bias) in enumerate(model['layers']):
                x=x@weight.T+bias
                if i<len(model['layers'])-1:
                    x=_silu(x)
            if model['kind']=='residual':
                weights=.5*np.tanh(x);weights[:,8]+=1
            else:
                weights=np.exp(x-x.max(axis=1,keepdims=True));weights/=weights.sum(axis=1,keepdims=True)
            results.append(np.einsum('nck,nk->nc',pack['experts']/scale,weights)*scale)
        return np.mean(results,axis=0)

    def __call__(self, history, now):
        pack=self.extractor.at(history,now)
        return pack['anchor']+self.displacement(pack)


class Expert:
    def __init__(self,dataset,index=8):
        self.extractor=CausalExperts(dataset);self.index=index

    def __call__(self,history,now):
        pack=self.extractor.at(history,now)
        return pack['anchor']+pack['experts'][:,:,self.index]


class Q1Transferred:
    """Fixed Q1 full-training point model, for official validation comparison.

    Nearest pre-fitted nominal horizon is selected using the disclosed sample
    age. Actual propagation still uses the full elapsed time. Ages beyond 1 s
    are explicit transfer extrapolation, not an additional fitted Q1 horizon.
    The dense Q1 radius/quality predictions are not used by this adapter.
    """
    def __init__(self, interface):
        self.interface=interface
        self.horizons=sorted(map(float,interface.models))

    def __call__(self,history,now):
        age=max(now-float(history.time[-1]),0.)
        if age<=1e-9:
            return history.centres[-1].copy()
        h=min(self.horizons,key=lambda value:abs(value-age))
        head=history.auxiliary['head']
        result,_=self.interface.point(history.time,history.points,history.bad,
                                     head['time'],head['position'],now,h)
        return result
