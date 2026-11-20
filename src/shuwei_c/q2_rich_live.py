"""CPU-only disclosed-history adapter for the new independently trained family."""
from pathlib import Path
import numpy as np
import torch
from .q2_features import DATASETS,transform_features
from .q2_rich_features import RichExperts


class RichPredictor:
    def __init__(self,models,dataset):
        self.extractor=RichExperts(dataset);self.dataset=dataset;self.models=[]
        for source in models:
            model=torch.load(source,map_location='cpu',weights_only=False) if isinstance(source,(str,Path)) else source
            state={k:v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v) for k,v in model['state_dict'].items()}
            self.models.append(dict(mean=model['mean'],std=model['std'],scale=model['noise'][dataset]['hand_scale'],
                names=model['feature_names'],experts=len(model['expert_names']),knots=np.log1p(model['prior']['knots']),
                coefficients=np.asarray(model['prior']['coefficients'],np.float32),layers=[(state[f'{j}.weight'],state[f'{j}.bias']) for j in (0,2,4)]))
        if not self.models:raise ValueError('No rich checkpoint')

    def displacement(self,pack):
        predictions=[]
        for model in self.models:
            count=len(model['names']);assert pack['feature_names'][:count]==model['names']
            x=transform_features(pack['features'][:,:count],model['names'],model['scale'])
            x=np.clip((x-model['mean'])/model['std'],-10,10).astype(np.float32)
            for i,(weight,bias) in enumerate(model['layers']):
                x=x@weight.T+bias
                if i<2:x=x/(1+np.exp(-np.clip(x,-80,80)))
            age=np.log1p(np.clip(pack['age'],0.,np.expm1(model['knots'][-1])))
            right=np.searchsorted(model['knots'],age,side='right').clip(1,len(model['knots'])-1);left=right-1
            f=(age-model['knots'][left])/(model['knots'][right]-model['knots'][left]);offset=DATASETS.index(self.dataset)*len(model['knots'])
            c=model['coefficients'];base=((1-f[:,None])*c[offset+left]+f[:,None]*c[offset+right]).astype(np.float32)
            e=pack['experts'][:,:,:model['experts']]/np.float32(model['scale'])
            predictions.append(np.einsum('nck,nk->nc',e,base+.5*np.tanh(x))*np.float32(model['scale']))
        return np.mean(predictions,axis=0)

    def __call__(self,history,now):
        pack=self.extractor.at(history,now)
        return pack['anchor']+self.displacement(pack)
