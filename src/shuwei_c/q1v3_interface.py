"""Disclosed-prefix v3 point prediction and stateful calibrated-ball interface.

The specification explicitly identifies fitted checkpoints and calibration.
This module never loads a recording or looks up hidden observations.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from .q1v3_learning import network
from .q1v3_state import DATASETS
from .q1v3_prefix import causal_pack
from .q1v3_online import StreamingBallState


class Q1V3:
    def __init__(self,specification,dataset):
        self.spec=json.loads(Path(specification).read_text()) if isinstance(specification,(str,Path)) else specification
        self.dataset=dataset;self.scale=self.spec['noise'][dataset]['hand_scale'];self.models={}
        for h,paths in self.spec['point_checkpoints'].items():
            self.models[h]=[]
            for path in paths:
                m=torch.load(path,map_location='cpu',weights_only=False);net=network(len(m['feature_names']),len(m['base']['coefficients'][0]));net.load_state_dict(m['state_dict']);net.eval();self.models[h].append((m,net))
        cal=self.spec['calibration'];self.parameters=json.loads(Path(cal['parameters']).read_text());self.quantiles={}
        for level,path in cal['quantile_models'].items():self.quantiles[level]=joblib.load(path)
        floor=json.loads(Path(cal['flagged_floor_parameters']).read_text())['full_training']['quantiles'] if cal['flagged_floor'] else None
        self.ledger=StreamingBallState(self.scale,self.spec['levels'],cal['feedback_rate'],floor)

    def point(self,time,points,bad,head_time,head_position,query_time,nominal_horizon):
        pack=causal_pack(time,points,bad,head_time,head_position,self.dataset,self.scale,query_time,nominal_horizon)
        key=str(float(nominal_horizon));predictions=[]
        for m,net in self.models[key]:
            if pack['feature_names']!=m['feature_names']:raise ValueError('Feature schema differs from trained checkpoint')
            x=torch.tensor(np.clip((pack['features']-m['mean'])/m['std'],-10,10),dtype=torch.float32)
            base=np.asarray(m['base']['coefficients'],np.float32);base=base[DATASETS.index(self.dataset)] if m['base']['dataset_specific'] else base[0]
            with torch.inference_mode():coef=base[None,:]+.5*torch.tanh(net(x)).numpy()
            predictions.append(np.einsum('nck,nk->nc',pack['experts'],coef))
        delta=np.mean(predictions,axis=0)*self.scale
        return pack['centres']+delta,pack

    def forecast(self,time,points,bad,head_time,head_position,query_time,nominal_horizon=.1):
        position,pack=self.point(time,points,bad,head_time,head_position,query_time,nominal_horizon)
        result=dict(position=position,origin_s=pack['origin_s'],query_s=pack['query_s'],nominal_horizon=nominal_horizon,
                    version=self.spec['version'],radius=None,forecast_id=None)
        if nominal_horizon!=self.spec['calibration']['horizon']:return result
        rows={f'x_{name}':pack['features'][:,i] for i,name in enumerate(pack['feature_names'])};rows.update(self.ledger.features(pack['origin_s']));frame=pd.DataFrame(rows);base=[]
        for level in self.spec['levels']:
            key=f'quantile_online_leaves15_{level}';entry=self.parameters[key];model=self.quantiles[str(level)]
            raw=np.maximum(np.expm1(model.predict(frame[entry['columns']])),1e-5)
            shift=entry['correction']['dataset_shift'][self.dataset];base.append(np.maximum(raw+shift,1e-5))
        key,radius=self.ledger.issue(pack['origin_s'],query_time,position,np.stack(base,axis=1),np.asarray(bad)[-1])
        result.update(radius=radius,forecast_id=key,levels=self.spec['levels'],calibration_scope='Dense disclosed Q1 histories; sparse/hidden histories require separate calibration')
        return result

    def observe(self,forecast_id,reference_time,points,bad,arrived_at=None):
        points=np.asarray(points);position=points.mean(axis=1) if points.shape==(2,21,3) else points
        self.ledger.observe(forecast_id,reference_time,position,~np.asarray(bad,bool),arrived_at)
