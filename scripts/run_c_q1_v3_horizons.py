#!/usr/bin/env python3
"""Transfer the fixed 0.1-second gate recipe to the four other declared horizons.

AI-assisted by Codex, 2026-11-20. No per-horizon validation tuning or test data.
"""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.data import Dataset
from shuwei_c.motion import estimate_noise
from shuwei_c.q1v3_state import pack_records,fit_distance,score
from shuwei_c.q1v3_learning import fit_gate,predict_gate


def main():
    out=Path('artifacts/c_q1_v3/horizon_transfer');out.mkdir(exist_ok=True);started=time.perf_counter();data=Dataset();train=data.records('train');validation=data.records('validation')
    noise=estimate_noise(train);config=json.loads(Path('configs/c_q1_v2.json').read_text());v2=json.loads(Path('artifacts/c_q1_v2/state_models_development.json').read_text())['horizons'];torch.cuda.set_per_process_memory_fraction(.2)
    rows=[];diagnostics=[]
    for horizon in [1/30,.2,.5,1.]:
        a=pack_records(train,horizon,config,noise);b=pack_records(validation,horizon,config,noise);base=fit_distance(a,.01,True);pred=[]
        for seed in [20261120,20261121,20261122]:
            model=fit_gate(a,base,10,.001,seed);torch.save(model,out/f'gate_h{horizon:.6f}_s{seed}.pt');pred.append(predict_gate(b,model))
        family=v2[str(horizon)]['family_models'];e=b['experts'];predictions=dict(gate=np.mean(pred,axis=0),linear=e[...,0],scalar=e[...,0]*family['scalar_damped']['coefficients'][0],multiscale=np.einsum('nck,k->nc',e[...,[0,2,4,6]],family['multiscale']['coefficients']))
        f=b['meta'].copy()
        for name,p in predictions.items():f[name+'_error']=np.linalg.norm(p-b['target'],axis=1)*f.scale
        f.to_parquet(out/f'validation_h{horizon:.6f}.parquet',index=False)
        for ds in ['ALL','EgoDemo','EgoPro','EgoStandard']:
            g=f if ds=='ALL' else f[f.dataset==ds]
            for name in predictions:rows.append(dict(horizon=horizon,dataset=ds,method=name,normalized_error=score(g,g[name+'_error']/g.scale),mean_error=score(g,g[name+'_error']),n=len(g)))
        diagnostics.append(dict(horizon=horizon,base=base));pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
        print(json.dumps(dict(horizon=horizon,gate=rows[-4]['mean_error'],elapsed_s=time.perf_counter()-started)),flush=True)
    (out/'protocol.json').write_text(json.dumps(dict(horizons=[1/30,.2,.5,1.],noise=noise,diagnostics=diagnostics,test_used=False,elapsed_s=time.perf_counter()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
