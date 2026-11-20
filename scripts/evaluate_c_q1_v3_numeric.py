#!/usr/bin/env python3
"""Evaluate the fixed numerical v3 bundle; never fit from evaluation outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
import torch
from shuwei_c.data import Dataset
from shuwei_c.q1v3_state import pack_records,score
from shuwei_c.q1v3_learning import predict_gate
from shuwei_c.q1v3_calibration import mature_residual_features,radius_metrics
from shuwei_c.q1v3_feedback import adapt_radii


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['validation','test'],required=True);args=parser.parse_args()
    root=Path('artifacts/c_q1_v3/frozen_numeric');freeze=json.loads((root/'freeze.json').read_text());out=root/args.split;out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('This fixed-version split was already evaluated')
    for p,h in {**freeze['inherited_v2_hashes'],**freeze['source_hashes']}.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,f'Frozen input changed: {p}'
    started=datetime.now(timezone.utc).isoformat();torch.set_num_threads(2);config=json.loads(Path('configs/c_q1_v2.json').read_text());records=Dataset().records(args.split)
    v2=json.loads(Path('artifacts/c_q1_v2/state_models_development.json').read_text())['horizons'];rows=[];radius_rows=[]
    for horizon,paths in freeze['point_checkpoints'].items():
        h=float(horizon);pack=pack_records(records,h,config,freeze['noise']);models=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
        e=pack['experts'];old=v2[horizon]['family_models'];predictions=dict(gate=np.mean([predict_gate(pack,m) for m in models],axis=0),linear=e[...,0],
            scalar=e[...,0]*old['scalar_damped']['coefficients'][0],multiscale=np.einsum('nck,k->nc',e[...,[0,2,4,6]],old['multiscale']['coefficients']))
        f=pack['meta'].copy()
        for k,name in enumerate(pack['feature_names']):f['x_'+name]=pack['features'][:,k]
        for name,p in predictions.items():
            f[name+'_normalized_error']=np.linalg.norm(p-pack['target'],axis=1);f[name+'_error']=f[name+'_normalized_error']*f.scale
        for ds in ['ALL','EgoDemo','EgoPro','EgoStandard']:
            g=f if ds=='ALL' else f[f.dataset==ds]
            for name in predictions:rows.append(dict(horizon=h,dataset=ds,method=name,n=len(g),environments=g.environment.nunique(),mean_error=score(g,g[name+'_error']),normalized_error=score(g,g[name+'_normalized_error'])))
        if h==freeze['calibration']['horizon']:
            f=mature_residual_features(f);cal=freeze['calibration'];parameters=json.loads(Path(cal['parameters']).read_text());base=[]
            for level in freeze['levels']:
                entry=parameters[f'quantile_online_leaves15_{level}'];model=joblib.load(cal['quantile_models'][str(level)])
                raw=np.maximum(np.expm1(model.predict(f[entry['columns']])),1e-5);shift=np.array([entry['correction']['dataset_shift'][ds] for ds in f.dataset]);base.append(np.maximum(raw+shift,1e-5))
            base=np.maximum.accumulate(np.stack(base,axis=1),axis=1)
            if cal['flagged_floor']:
                floor=json.loads(Path(cal['flagged_floor_parameters']).read_text())['full_training']['quantiles'];mask=f.x_flag_now.to_numpy()>0;base[mask]=np.maximum(base[mask],floor)
            radius,_=adapt_radii(f,base,freeze['levels'],cal['feedback_rate']);np.savez_compressed(out/'main_radii.npz',radii=radius)
            for condition in ['all','current_good','current_flagged']:
                mask=np.ones(len(f),bool) if condition=='all' else ((f.x_flag_now.to_numpy()>0) if condition=='current_flagged' else f.x_flag_now.to_numpy()==0)
                for k,level in enumerate(freeze['levels']):
                    metrics=radius_metrics(f.loc[mask].reset_index(drop=True),radius[mask,k],level,'conditional_feedback_flag_floor')
                    for item in metrics:item.update(horizon=h,condition=condition)
                    radius_rows+=metrics
        f.to_parquet(out/f'points_h{h:.6f}.parquet',index=False)
    pd.DataFrame(rows).to_csv(out/'point_metrics.csv',index=False);pd.DataFrame(radius_rows).to_csv(out/'radius_metrics.csv',index=False)
    manifest=dict(split=args.split,started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),freeze_timestamp=freeze['timestamp_utc'],
        freeze_sha256=hashlib.sha256((root/'freeze.json').read_bytes()).hexdigest(),records=len(records),environments=len({r.environment for r in records}),
        interpretation='Fixed-version reevaluation of the already-observed official test' if args.split=='test' else 'Frozen numerical bundle reproduction of development validation; no fitting',fitted_from_this_split=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))


if __name__=='__main__':main()
