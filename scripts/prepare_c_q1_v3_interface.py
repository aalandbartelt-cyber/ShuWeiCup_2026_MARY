#!/usr/bin/env python3
"""Prepare an explicitly unfrozen interface specification and replay dense prefixes."""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.data import Dataset
from shuwei_c.q1v3_interface import Q1V3
from shuwei_c.q1v3_learning import predict_gate
from shuwei_c.q1v3_state import pack_records


def main():
    out=Path('artifacts/c_q1_v3/interface');out.mkdir(exist_ok=True);torch.set_num_threads(2);start=time.perf_counter();seeds=[20261120,20261121,20261122]
    noise=json.loads(Path('artifacts/c_q1_v3/horizon_transfer/protocol.json').read_text())['noise'];levels=[.5,.8,.9,.95]
    spec=dict(version='q1-v3-development-unfrozen',test_previously_observed=True,levels=levels,noise=noise,
        point_checkpoints={str(h):[f'artifacts/c_q1_v3/state_screen/gate_r0.001_seed{s}_e10_0.1.pt' if h==.1 else f'artifacts/c_q1_v3/horizon_transfer/gate_h{h:.6f}_s{s}.pt' for s in seeds] for h in [1/30,.1,.2,.5,1.]},
        calibration=dict(horizon=.1,parameters='artifacts/c_q1_v3/calibration_screen/parameters.json',feedback_rate=.01,flagged_floor=True,
            flagged_floor_parameters='artifacts/c_q1_v3/flagged_calibration/parameters.json',quantile_models={str(p):f'artifacts/c_q1_v3/calibration_screen/quantile_online_leaves15_{p}.joblib' for p in levels}))
    (out/'development_specification.json').write_text(json.dumps(spec,indent=2)+'\n');data=Dataset();rows=[];config=json.loads(Path('configs/c_q1_v2.json').read_text())
    # Fixed early/central/late rows from each validation record; entire arrays
    # passed to the interface are explicit prefixes, never source identities.
    for rec in data.records('validation'):
        t,p,bad=rec.hands();head=rec.streams['head'];batch=pack_records([rec],.1,config,noise)
        predictions=np.mean([predict_gate(batch,torch.load(path,map_location='cpu',weights_only=False)) for path in spec['point_checkpoints']['0.1']],axis=0)
        interface=Q1V3(spec,rec.dataset)
        for origin in np.unique(batch['meta'].origin)[np.linspace(0,len(np.unique(batch['meta'].origin))-1,3).astype(int)]:
            m=batch['meta'];ix=np.flatnonzero(m.origin==origin);query=m.target_s.iloc[ix[0]];headix=head['time']<=t[origin]
            result=interface.forecast(t[:origin+1],p[:origin+1],bad[:origin+1],head['time'][headix],head['position'][headix],query)
            for row in ix:
                hand=int(m.hand.iloc[row]);reference=p[origin,hand].mean(axis=0)+predictions[row]*m.scale.iloc[row]
                discrepancy=float(np.linalg.norm(reference-result['position'][hand]));assert discrepancy<1e-6
                rows.append(dict(episode=rec.episode,origin=int(origin),hand=hand,point_discrepancy=discrepancy,radius_nested=bool(np.all(np.diff(result['radius'][hand])>=0))))
            # This sparse sample of interface calls is not itself a coverage
            # experiment. Expire queries; do not fabricate intervening returns.
            interface.ledger.discard(result['forecast_id'])
    pd.DataFrame(rows).to_csv(out/'prefix_point_parity.csv',index=False)
    (out/'verification.json').write_text(json.dumps(dict(status='PASS',rows=len(rows),max_point_discrepancy=max(r['point_discrepancy'] for r in rows),dense_coverage_replay_pending=True,sparse_calibration_pending=True,test_used=False,elapsed_s=time.perf_counter()-start,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_interface.py'),Path('src/shuwei_c/q1v3_online.py')]}),indent=2)+'\n')


if __name__=='__main__':main()
