#!/usr/bin/env python3
"""Empirical correction for the declared sparse/delayed interface probe.

AI-assisted by Codex, 2026-11-20. Point models leave the calibration environment
out. This does not certify arbitrary Q2 policies or observation patterns.
"""
from pathlib import Path
import copy,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset,balanced_weights
from shuwei_c.q1v2_state import PrefixPredictor,weighted_quantile


def main():
    out=Path('artifacts/c_q1_v2');config=json.loads(Path('configs/c_q1_v2.json').read_text());full=json.loads((out/'state_models_development.json').read_text());rows=[];records=Dataset().records('train')
    for n,rec in enumerate(records):
        models=copy.deepcopy(full)
        for entry in models['horizons'].values():
            fold=next(f for f in entry['fold_specs'] if f['held_environment']==rec.environment)
            entry['selected']=fold['spec'];models['noise']=fold['noise']
        predictor=PrefixPredictor(models,config);t,p,bad=rec.hands();c=p.mean(axis=2)
        for fraction in [.2,.5,.8]:
            start=int(np.searchsorted(t,t[0]+fraction*(t[-1]-t[0])))
            if start<14 or t[start]+1>t[-1]:continue
            stop=int(np.searchsorted(t,t[start]+1,side='right'))
            for interval in [.2,.5]:
                for delay in [0.,.1]:
                    received=list(range(start));pending=[];next_request=t[start]
                    for j in range(start,stop):
                        now=t[j]
                        if now+1e-9>=next_request:pending.append((j,now+delay));next_request+=interval
                        ready=[entry for entry in pending if entry[1]<=now+1e-9]
                        received.extend(entry[0] for entry in ready);pending=[entry for entry in pending if entry not in ready]
                        ix=np.array(received);prediction=predictor.forecast(t[ix],c[ix],bad[ix],now,rec.dataset)
                        for hand in range(2):
                            if bad[j,hand]:continue
                            error=float(np.linalg.norm(prediction['position'][hand]-c[j,hand]))
                            rows.append(dict(episode=rec.episode,environment=rec.environment,hand=hand,interval=interval,delay=delay,error=error,radius=prediction['radius'][hand]))
        print(f'Calibrated record {n+1}/{len(records)}',flush=True)
    f=pd.DataFrame(rows);f.to_parquet(out/'q2_sparse_calibration_residuals.parquet',index=False)
    w=balanced_weights(f.environment,f.episode);factor=weighted_quantile(f.error/f.radius,w,.9)
    protocol=dict(nominal=.9,multiplier=factor,calibration_environments=f.environment.nunique(),train_records=f.episode.nunique(),
        fitting='point forecasts leave environment out; empirical residual correction of development Q1 radius; equal environment/record weights',
        scope='only tested periodic 0.2/0.5 s responses, delay 0/0.1 s; no arbitrary-policy coverage guarantee',test_used=False)
    (out/'q2_sparse_calibration.json').write_text(json.dumps(protocol,indent=2)+'\n')
    validation=pd.read_parquet(out/'q2_interface_replay.parquet');g=validation[validation.method=='selected'].copy()
    summary=[]
    for scheme,multiplier in [('dense_Q1',1.),('sparse_recalibrated',factor)]:
        g['evaluated_radius']=g.radius*multiplier;g['covered']=(g.error<=g.evaluated_radius).astype(float)
        s=g.groupby(['interval','delay','environment','episode','hand'])[['error','evaluated_radius','covered']].mean().groupby(
            ['interval','delay','environment','episode']).mean().groupby(['interval','delay','environment']).mean().groupby(['interval','delay']).mean().reset_index()
        s['scheme']=scheme;summary.append(s)
    pd.concat(summary).to_csv(out/'q2_sparse_coverage.csv',index=False);print(json.dumps(protocol),flush=True)


if __name__=='__main__':main()
