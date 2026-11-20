#!/usr/bin/env python3
"""Shared-hand neural probability corrections from articulated image histories.

AI-assisted by Codex, 2026-11-20. Causal visual descriptors; training transition
priors are environment-OOF. All screened candidates retained, no test data.
"""
from pathlib import Path
import json,sys,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from shuwei_c import q1v2_quality as q
from shuwei_c.q1v3_shared_risk import features
from shuwei_c.q1v3_landmarks import NAMES


def select_visual(array,kind):
    n=len(NAMES);stride=3*n+6
    indices=[]
    for start in [0,stride]:
        if kind=='scalar':indices+=list(range(start,start+12))
        elif kind=='shape':indices+=list(range(start,start+n))
        elif kind=='motion':indices+=list(range(start,start+stride))
    return array[:,:,indices]


def main():
    source=Path('artifacts/c_q1_v3/articulated_features');assert (source/'manifest.json').exists();out=Path('artifacts/c_q1_v3/articulated_risk');out.mkdir(exist_ok=True);start=time.perf_counter()
    frames={s:pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_{s}_0.1.parquet') for s in ['train','validation']}
    visual={s:np.load(source/f'quality_{s}.npz')['features'] for s in frames};numeric={};flags={}
    for s,f in frames.items():numeric[s],flags[s],_=features(f,None,False)
    prior={'train':np.zeros((len(frames['train']),4))}
    for fi,hi in GroupKFold(4).split(frames['train'],groups=frames['train'].environment):prior['train'][hi]=q.predict_baseline(q.fit_baseline(frames['train'].iloc[fi]),frames['train'].iloc[hi])
    prior['validation']=q.predict_baseline(q.fit_baseline(frames['train']),frames['validation']);weights=q.weights(frames['train']);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    lp={s:torch.tensor(np.log(np.clip(p,1e-6,1)),device='cuda',dtype=torch.float32) for s,p in prior.items()};target=frames['train'].target_state.to_numpy()
    y=torch.tensor(np.c_[target>0,target%2,target//2],device='cuda',dtype=torch.float32);w=torch.tensor(weights*len(weights),device='cuda',dtype=torch.float32);lw=torch.tensor([.5,.25,.25],device='cuda')
    rows=q.metrics(frames['validation'],prior['validation'],'transition');predictions={'transition':prior['validation']}
    for kind in ['numeric','scalar','shape','motion']:
        raw={s:np.concatenate([numeric[s],flags[s][...,None],*([] if kind=='numeric' else [select_visual(visual[s],kind)])],axis=-1) for s in frames}
        mean=np.average(raw['train'].mean(axis=1),weights=weights,axis=0);std=np.maximum(np.sqrt(np.average(((raw['train']-mean)**2).mean(axis=1),weights=weights,axis=0)),1e-4)
        x={s:torch.tensor(np.clip((a-mean)/std,-4,4),device='cuda',dtype=torch.float32) for s,a in raw.items()}
        for reg in [.001,.01]:
            for seed in [20261120,20261121,20261122]:
                torch.manual_seed(seed);net=torch.nn.Sequential(torch.nn.Linear(x['train'].shape[-1],32),torch.nn.SiLU(),torch.nn.Dropout(.3),torch.nn.Linear(32,1)).cuda();torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.05)
                def probability(xx,base):
                    delta=2*torch.tanh(net(xx).squeeze(-1));logits=torch.stack([torch.zeros(len(xx),device='cuda'),delta[:,0],delta[:,1],delta.sum(dim=1)],dim=1)
                    return torch.softmax(base+logits,dim=1),delta
                for epoch in range(1,21):
                    net.train()
                    for ix in torch.randperm(len(w),device='cuda').split(4096):
                        p,delta=probability(x['train'][ix],lp['train'][ix]);prob=torch.stack([1-p[:,0],p[:,1]+p[:,3],p[:,2]+p[:,3]],dim=1)
                        loss=((((prob-y[ix])**2)*lw).sum(dim=1)*w[ix]).mean()+reg*delta.square().mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);optimizer.step()
                    if epoch in [5,10,20]:
                        name=f'{kind}_r{reg}_s{seed}_e{epoch}';net.eval()
                        with torch.inference_mode():p=probability(x['validation'],lp['validation'])[0].cpu().numpy()
                        predictions[name]=p;metrics=q.metrics(frames['validation'],p,name)
                        for row in metrics:row.update(variant=kind,regularization=reg,seed=seed,epoch=epoch)
                        rows+=metrics;pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
                        torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},mean=mean,std=std,kind=kind,regularization=reg,seed=seed,epoch=epoch),out/(name+'.pt'))
                result=next(r['brier'] for r in metrics if r['dataset']=='ALL' and r['condition']=='all' and r['target']=='any');print(json.dumps(dict(variant=kind,regularization=reg,seed=seed,brier=result,seconds=time.perf_counter()-start)),flush=True)
    np.savez_compressed(out/'probabilities.npz',**predictions);(out/'manifest.json').write_text(json.dumps(dict(test_used=False,elapsed_s=time.perf_counter()-start,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
