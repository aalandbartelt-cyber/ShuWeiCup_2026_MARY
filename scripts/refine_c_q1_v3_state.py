#!/usr/bin/env python3
"""Refine the unchanged convex initializer and assess its numerical impact."""
from pathlib import Path
import json,sys,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.q1v3_refine import refine
from shuwei_c.q1v3_state import predict_distance,score
from shuwei_c.q1v3_learning import fit_gate,predict_gate


def main():
    out=Path('artifacts/c_q1_v3/refined_state');out.mkdir(exist_ok=True);source=Path('artifacts/c_q1_v3/state_screen');start=time.perf_counter();packs={}
    for split in ['train','validation']:
        p=dict(np.load(source/f'{split}_0.1.npz'));p['meta']=pd.read_parquet(source/f'{split}_0.1_meta.parquet');p['feature_names']=json.loads((source/'feature_names.json').read_text());packs[split]=p
    initial=json.loads(Path('artifacts/c_q1_v3/numeric_oof/protocol.json').read_text())['base_config'];model=refine(packs['train'],initial)
    (out/'initializer.json').write_text(json.dumps(model,indent=2)+'\n');print(json.dumps(dict(before=initial['objective'],after=model['objective'],converged=model['convergence'],gradient=model['projected_gradient_inf'],seconds=time.perf_counter()-start)),flush=True)
    b=packs['validation'];predictions={'initial_linear':predict_distance(b,initial),'refined_linear':predict_distance(b,model)};pred=[];torch.cuda.set_per_process_memory_fraction(.25)
    for seed in [20261120,20261121,20261122]:
        gate=fit_gate(packs['train'],model,10,.001,seed);torch.save(gate,out/f'gate_s{seed}.pt');pred.append(predict_gate(b,gate))
    predictions['refined_gate']=np.mean(pred,axis=0);rows=[]
    for name,p in predictions.items():
        error=np.linalg.norm(p-b['target'],axis=1)
        for ds in ['ALL','EgoDemo','EgoPro','EgoStandard']:
            mask=np.ones(len(error),bool) if ds=='ALL' else b['meta'].dataset.to_numpy()==ds
            rows.append(dict(method=name,dataset=ds,error=score(b['meta'].loc[mask],error[mask]),mean_error=score(b['meta'].loc[mask],error[mask]*b['meta'].loc[mask,'scale'])))
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False);np.savez_compressed(out/'predictions.npz',**predictions)
    (out/'manifest.json').write_text(json.dumps(dict(elapsed_s=time.perf_counter()-start,test_used=False,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_refine.py')]}),indent=2)+'\n')


if __name__=='__main__':main()
