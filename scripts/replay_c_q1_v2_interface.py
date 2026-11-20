#!/usr/bin/env python3
"""Q1 -> Q2 integration probe using actual delayed point observations only.

AI-assisted by Codex, 2026-11-20. This is a representative integration probe,
not a replacement for the full Q2 policy/budget benchmark.
"""
from pathlib import Path
import copy,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset
from shuwei_c.q1v2_state import PrefixPredictor
from shuwei_c.q1v2_quality import dataset_frame,columns_for


def main():
    out=Path('artifacts/c_q1_v2');config=json.loads(Path('configs/c_q1_v2.json').read_text());models=json.loads((out/'state_models_development.json').read_text());predictor=PrefixPredictor(models,config)
    rows=[];ledger=[];checks=[];data=Dataset()
    for rec in data.records('validation'):
        t,p,bad=rec.hands();c=p.mean(axis=2)
        # Actual full-record perturbation audit: features before the cut must not
        # change, including as-of flags/head/camera positions, not just slicing c.
        frame=dataset_frame([rec],.1,config);cut=t[len(t)//2];changed=copy.deepcopy(rec)
        for stream in changed.streams.values():
            future=stream['time']>cut
            if 'position' in stream:stream['position'][future]+=731.
            if 'bad' in stream:stream['bad'][future]=~stream['bad'][future]
        altered=dataset_frame([changed],.1,config);cols=columns_for(frame,'geometry');mask=frame.time_s<=cut
        np.testing.assert_array_equal(frame.loc[mask,cols].to_numpy(),altered.loc[mask,cols].to_numpy())
        checks.append(dict(episode=rec.episode,checked_rows=int(mask.sum()),columns=len(cols),future_perturbation_pass=True))
        # Three fixed locations, one second masked interval. History before it
        # is available; later samples are delivered individually after latency.
        for fraction in [.2,.5,.8]:
            start=int(np.searchsorted(t,t[0]+fraction*(t[-1]-t[0])))
            if start<14 or t[start]+1>t[-1]:continue
            stop=int(np.searchsorted(t,t[start]+1,side='right'))
            for interval in [.2,.5]:
                for delay in [0.,.1]:
                    received=list(range(start));pending=[];next_request=t[start]
                    for j in range(start,stop):
                        now=t[j]
                        if now+1e-9>=next_request:
                            pending.append((j,now+delay));next_request+=interval
                            ledger.append(dict(episode=rec.episode,fraction=fraction,interval=interval,delay=delay,source_index=j,request_s=now,return_s=now+delay))
                        ready=[entry for entry in pending if entry[1]<=now+1e-9]
                        received.extend(entry[0] for entry in ready);pending=[entry for entry in pending if entry not in ready]
                        ix=np.array(received);assert t[ix[-1]]<=now+1e-9
                        prediction=predictor.forecast(t[ix],c[ix],bad[ix],now,rec.dataset)
                        age=now-t[ix[-1]];linear=c[ix[-1]]+age*(c[ix[-1]]-c[ix[-2]])/(t[ix[-1]]-t[ix[-2]])
                        for method,position in [('constant',c[ix[-1]]),('linear',linear),('selected',prediction['position'])]:
                            for hand in range(2):
                                if bad[j,hand]:continue
                                error=float(np.linalg.norm(position[hand]-c[j,hand]))
                                rows.append(dict(episode=rec.episode,environment=rec.environment,dataset=rec.dataset,fraction=fraction,
                                    interval=interval,delay=delay,time_s=now,latest_received_s=t[ix[-1]],hand=hand,method=method,error=error,
                                    radius=prediction['radius'][hand] if method=='selected' else np.nan,
                                    covered=float(error<=prediction['radius'][hand]) if method=='selected' else np.nan))
    frame=pd.DataFrame(rows);frame.to_parquet(out/'q2_interface_replay.parquet',index=False)
    summary=frame.groupby(['method','interval','delay','environment','episode','hand'])[['error','radius','covered']].mean().groupby(
        ['method','interval','delay','environment','episode']).mean().groupby(['method','interval','delay','environment']).mean().groupby(['method','interval','delay']).mean()
    summary.to_csv(out/'q2_interface_summary.csv');pd.DataFrame(ledger).to_csv(out/'q2_interface_ledger.csv',index=False);pd.DataFrame(checks).to_csv(out/'real_record_feature_causality.csv',index=False)
    (out/'q2_interface_protocol.json').write_text(json.dumps(dict(windows='one second at 20/50/80 percent of each validation record',
        intervals=[.2,.5],delays=[0.,.1],responses='a single sample at REQUEST time, delivered after delay; intervening hidden values never disclosed',
        limitations=['no Q2 policy selection','dense-history Q1 calibration is diagnostic only after sparse updates','same request budget within each comparison'],test_used=False),indent=2)+'\n')
    print(summary.to_string())


if __name__=='__main__':main()
