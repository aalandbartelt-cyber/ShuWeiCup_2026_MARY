"""Frozen, versioned Q1 state/risk interface for the Q2 integration layer.

AI-assisted by Codex, 2026-11-20. It accepts only caller-disclosed observations.
"""
from pathlib import Path
import hashlib,json
import joblib
import numpy as np
import pandas as pd
from .q1v2_state import PrefixPredictor
from .q1v2_quality import predict_baseline,predict_classifier,temperature


class FrozenQ1:
    def __init__(self,root='artifacts/c_q1_v2',config_path='configs/c_q1_v2.json'):
        root=Path(root);path=root/'freeze.json';self.freeze=json.loads(path.read_text())
        self.version=hashlib.sha256(path.read_bytes()).hexdigest();self.config=json.loads(Path(config_path).read_text())
        self.state=PrefixPredictor(self.freeze['state'],self.config)
        self.sparse=json.loads((root/'q2_sparse_calibration.json').read_text())
        self.quality={h:joblib.load(s['model_path']) for h,s in self.freeze['quality'].items()}

    def forecast(self,time,centres,bad,query_time,dataset,response_pattern=None):
        result=self.state.forecast(time,centres,bad,query_time,dataset)
        result['model_sha256']=self.version;result['calibration_scope']='Q1 dense observed history; empirical marginal ball'
        if response_pattern is not None:
            interval,delay=response_pattern
            if interval not in [.2,.5] or delay not in [0.,.1]:raise ValueError('Sparse pattern has no evaluated calibration')
            result['radius']=result['radius']*self.sparse['multiplier']
            result['calibration_scope']='Declared sparse periodic probe only; validation undercoverage remains'
        return result

    def flag_risk(self,last_bad,dataset,horizon=.1,causal_features=None):
        key=str(float(horizon));entry=self.freeze['quality'][key];spec=entry['spec']
        if spec['kind']=='baseline':
            state=int(bool(last_bad[0]))+2*int(bool(last_bad[1]))
            frame=pd.DataFrame([dict(dataset=dataset,current_state=state)])
            p=predict_baseline(self.quality[key],frame)
        else:
            if causal_features is None:raise ValueError('Selected classifier requires the declared causal feature row')
            frame=pd.DataFrame([causal_features])[spec['columns']]
            p=predict_classifier(self.quality[key],frame,spec['columns'])
        p=temperature(p,spec['temperature'])[0]
        return dict(joint_states=p,left=float(p[1]+p[3]),right=float(p[2]+p[3]),any=float(1-p[0]),
            horizon=horizon,target='supplied bad flag at future endpoint',model_sha256=self.version)
