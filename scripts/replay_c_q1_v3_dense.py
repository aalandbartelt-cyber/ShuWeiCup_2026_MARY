#!/usr/bin/env python3
"""Complete dense validation replay through the disclosed-prefix public API."""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.data import Dataset
from shuwei_c.q1v2_state import matched_targets
from shuwei_c.q1v3_interface import Q1V3


def main():
    root=Path('artifacts/c_q1_v3');out=root/'interface/dense_validation';out.mkdir(parents=True,exist_ok=True)
    if (out/'verification.json').exists():raise RuntimeError('Completed dense replay already exists')
    frozen=root/'frozen_numeric/freeze.json';spec=json.loads(frozen.read_text());config=json.loads(Path('configs/c_q1_v2.json').read_text());torch.set_num_threads(2);start=time.perf_counter();rows=[];logs=[]
    for rec in Dataset().records('validation'):
        t,p,bad=rec.hands();head=rec.streams['head'];origins,targets=matched_targets(t,.1,config['warmup_s'],config['target_tolerance_steps']);api=Q1V3(spec,rec.dataset);pending=[];returned=0
        for origin,target in zip(origins,targets):
            ready=[entry for entry in pending if entry[0]<=origin]
            for j,key in ready:api.observe(key,t[j],p[j],bad[j]);pending.remove((j,key));returned+=1
            hi=np.searchsorted(head['time'],t[origin],side='right')
            result=api.forecast(t[:origin+1],p[:origin+1],bad[:origin+1],head['time'][:hi],head['position'][:hi],t[target])
            pending.append((int(target),result['forecast_id']))
            for hand in range(2):
                if bad[target,hand]:continue
                error=float(np.linalg.norm(result['position'][hand]-p[target,hand].mean(axis=0)))
                row=dict(episode=rec.episode,environment=rec.environment,dataset=rec.dataset,origin=int(origin),target=int(target),hand=hand,error=error)
                for k,level in enumerate(spec['levels']):row[f'radius_{level}']=float(result['radius'][hand,k])
                rows.append(row)
        logs.append(dict(episode=rec.episode,issued_forecasts=len(origins),explicit_return_calls=returned,pending_at_last_origin=len(pending),elapsed_s=time.perf_counter()-start));print(json.dumps(logs[-1]),flush=True)
    actual=pd.DataFrame(rows);expected=pd.read_parquet(root/'frozen_numeric/validation/points_h0.100000.parquet');radii=np.load(root/'frozen_numeric/validation/main_radii.npz')['radii']
    for k,level in enumerate(spec['levels']):expected[f'expected_radius_{level}']=radii[:,k]*expected.scale
    fields=['episode','origin','hand'];compare=actual.merge(expected[fields+['gate_error',*[f'expected_radius_{x}' for x in spec['levels']]]],on=fields,validate='one_to_one');assert len(compare)==len(actual)==len(expected)
    point_difference=float(np.max(abs(compare.error-compare.gate_error)));radius_difference=max(float(np.max(abs(compare[f'radius_{x}']-compare[f'expected_radius_{x}']))) for x in spec['levels'])
    actual.to_parquet(out/'predictions.parquet',index=False);(out/'records.json').write_text(json.dumps(logs,indent=2)+'\n')
    result=dict(status='PASS' if point_difference<1e-6 and radius_difference<1e-5 else 'REVIEW',rows=len(actual),point_error_max_difference=point_difference,radius_max_difference=radius_difference,scope='All matched validation origins issued, including forecasts whose future flags are bad; only already-arrived qualified references update the ledger. Final metrics compare the qualified targets.',test_used=False,elapsed_s=time.perf_counter()-start,freeze_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert result['status']=='PASS'


if __name__=='__main__':main()
