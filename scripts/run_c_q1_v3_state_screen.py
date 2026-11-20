#!/usr/bin/env python3
"""Direct Euclidean-loss expert fitting and a small causal gating network.

AI-assisted by Codex, 2026-11-20. First screening only; environment OOF validation
of promising candidates is mandatory before final selection and calibration.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.data import Dataset
from shuwei_c.q1v3_state import pack_records,fit_distance,predict_distance,score,row_weights


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--horizon',type=float,default=.1);args=parser.parse_args()
    out=Path('artifacts/c_q1_v3/state_screen');out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    config=json.loads(Path('configs/c_q1_v2.json').read_text());noise=json.loads(Path('artifacts/c_q1_v2/state_models_development.json').read_text())['noise'];data=Dataset();packs={}
    for split in ['train','validation']:
        pack=pack_records(data.records(split),args.horizon,config,noise);packs[split]=pack
        np.savez_compressed(out/f'{split}_{args.horizon:.1f}.npz',experts=pack['experts'],features=pack['features'],target=pack['target'])
        pack['meta'].to_parquet(out/f'{split}_{args.horizon:.1f}_meta.parquet',index=False)
        (out/'feature_names.json').write_text(json.dumps(pack['feature_names'],indent=2)+'\n')
    train,val=packs['train'],packs['validation'];rows=[];predictions={};specs={}
    def record(name,prediction,extra=None):
        error=np.linalg.norm(prediction-val['target'],axis=1);frame=val['meta'];value=score(frame,error)
        row=dict(method=name,horizon=args.horizon,score=value)
        for ds in frame.dataset.unique():
            take=(frame.dataset==ds).to_numpy();row[ds]=score(frame.loc[take],error[take])
        rows.append(row);predictions[name]=prediction.astype(np.float32)
        pd.DataFrame(rows).to_csv(out/f'scores_{args.horizon:.1f}.csv',index=False)
        print(json.dumps(row),flush=True)
        if extra is not None:specs[name]=extra
    record('two_point',val['experts'][...,0])
    for grouped in [False,True]:
        for penalty in [.001,.01,.1]:
            name=f'distance_{"dataset" if grouped else "shared"}_{penalty}';spec=fit_distance(train,penalty,grouped);record(name,predict_distance(val,spec),spec)
    # Prototype gate has invariant inputs and scalar expert weights, preserving
    # common 3D rotation equivariance. Direct Euclidean objective matches Q1.
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20)
    w=row_weights(train['meta']);mean=np.average(train['features'],weights=w,axis=0);std=np.sqrt(np.average((train['features']-mean)**2,weights=w,axis=0));std=np.maximum(std,1e-4)
    tensors={split:(torch.tensor(np.clip((p['features']-mean)/std,-10,10),device='cuda',dtype=torch.float32),torch.tensor(p['experts'],device='cuda')) for split,p in packs.items()}
    tx,te=tensors['train'];vx,ve=tensors['validation'];y=torch.tensor(train['target'],device='cuda');weights=torch.tensor(w*len(w),device='cuda',dtype=torch.float32)
    base_name=min(specs,key=lambda n:next(r['score'] for r in rows if r['method']==n));base=specs[base_name]
    from shuwei_c.q1v3_state import DATASETS
    def coefficients(meta):
        c=np.array(base['coefficients'],np.float32)
        return c[[DATASETS.index(d) for d in meta.dataset]] if base['dataset_specific'] else np.repeat(c,len(meta),axis=0)
    tb=torch.tensor(coefficients(train['meta']),device='cuda');vb=torch.tensor(coefficients(val['meta']),device='cuda')
    for regularization in [.001,.01]:
        for seed in [20261120,20261121,20261122]:
            torch.manual_seed(seed);net=torch.nn.Sequential(torch.nn.Linear(tx.shape[1],48),torch.nn.SiLU(),torch.nn.Linear(48,48),torch.nn.SiLU(),torch.nn.Linear(48,te.shape[-1])).cuda()
            torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
            optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
            for epoch in range(1,61):
                order=torch.randperm(len(tx),device='cuda');net.train()
                for batch in order.split(4096):
                    correction=.5*torch.tanh(net(tx[batch]));coef=tb[batch]+correction
                    prediction=torch.einsum('nck,nk->nc',te[batch],coef)
                    distance=torch.sqrt(((prediction-y[batch])**2).sum(dim=1)+1e-8)
                    loss=(distance*weights[batch]).mean()+regularization*(correction**2).mean()
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);optimizer.step()
                if epoch in [10,20,40,60]:
                    net.eval()
                    with torch.inference_mode():pv=torch.einsum('nck,nk->nc',ve,vb+.5*torch.tanh(net(vx))).cpu().numpy()
                    name=f'gate_r{regularization}_seed{seed}_e{epoch}';record(name,pv)
                    torch.save(dict(state_dict=net.cpu().state_dict(),mean=mean,std=std,base=base,feature_names=train['feature_names'],seed=seed,epoch=epoch,regularization=regularization),out/f'{name}_{args.horizon:.1f}.pt');net.cuda()
    np.savez_compressed(out/f'predictions_{args.horizon:.1f}.npz',**predictions)
    (out/f'linear_specs_{args.horizon:.1f}.json').write_text(json.dumps(specs,indent=2)+'\n')
    (out/f'manifest_{args.horizon:.1f}.json').write_text(json.dumps(dict(elapsed_s=time.perf_counter()-started,test_used=False,
        stage='validation screen, not final grouped model verification',source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_state.py')]}),indent=2)+'\n')


if __name__=='__main__':main()
