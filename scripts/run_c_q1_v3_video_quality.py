#!/usr/bin/env python3
"""Causal visual residual on a joint-quality transition prior, local GPU.

AI-assisted by Codex, 2026-11-20. Transition priors for training are environment
OOF. Frozen visual PCA uses only training videos. Validation is development.
"""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from shuwei_c import q1v2_quality as q
from shuwei_c.q1v3_visual import align_visual


def main():
    out=Path('artifacts/c_q1_v3/video_quality');out.mkdir(exist_ok=True);started=time.perf_counter()
    root=Path('artifacts/c_q1_v3/visual_dino15');pca=joblib.load('artifacts/c_q1_v3/visual_screen/pca_train_only.joblib')
    frames={s:pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_{s}_0.1.parquet') for s in ['train','validation']}
    cols=q.columns_for(frames['train'],'geometry');visual={}
    for split,frame in frames.items():
        file=out/f'visual_{split}.npz'
        if file.exists():visual[split]=dict(np.load(file));continue
        hands=[]
        for side in [0,1]:
            meta=frame[['episode','time_s']].rename(columns={'time_s':'origin_s'});meta['hand']=side
            v,names=align_visual(meta,root,pca);hands.append(v)
        visual[split]={k:np.concatenate([v[k] for v in hands],axis=1) for k in hands[0]}
        np.savez_compressed(file,**visual[split])
    prior={};prior['train']=np.zeros((len(frames['train']),4))
    for fi,hi in GroupKFold(4).split(frames['train'],groups=frames['train'].environment):
        prior['train'][hi]=q.predict_baseline(q.fit_baseline(frames['train'].iloc[fi]),frames['train'].iloc[hi])
    prior['validation']=q.predict_baseline(q.fit_baseline(frames['train']),frames['validation'])
    weights=q.weights(frames['train']);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
    tensor_prior={s:torch.tensor(np.log(np.clip(p,1e-6,1)),device='cuda',dtype=torch.float32) for s,p in prior.items()}
    target=frames['train'].target_state.to_numpy();y=torch.tensor(np.c_[target>0,target%2,target//2],device='cuda',dtype=torch.float32)
    w=torch.tensor(weights*len(weights),device='cuda',dtype=torch.float32);rows=[];predictions={'transition':prior['validation']}
    rows+=q.metrics(frames['validation'],prior['validation'],'transition')
    variants={'numeric':[], 'diagnostics':['diagnostics'],'appearance':['appearance'],'visual_motion':['diagnostics','motion'],'all_visual':['diagnostics','appearance','motion']}
    for variant,keys in variants.items():
        feature={s:np.concatenate([f[cols].to_numpy(),*[visual[s][k] for k in keys]],axis=1) for s,f in frames.items()}
        mean=np.average(feature['train'],weights=weights,axis=0);std=np.maximum(np.sqrt(np.average((feature['train']-mean)**2,weights=weights,axis=0)),1e-4)
        x={s:torch.tensor(np.clip((a-mean)/std,-8,8),device='cuda',dtype=torch.float32) for s,a in feature.items()}
        for penalty in [.0001,.001]:
            for seed in [20261120,20261121,20261122]:
                torch.manual_seed(seed);net=torch.nn.Sequential(torch.nn.Linear(x['train'].shape[1],48),torch.nn.SiLU(),torch.nn.Dropout(.2),torch.nn.Linear(48,24),torch.nn.SiLU(),torch.nn.Linear(24,4)).cuda()
                torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.05)
                for epoch in range(1,41):
                    net.train()
                    for ix in torch.randperm(len(w),device='cuda').split(4096):
                        delta=2*torch.tanh(net(x['train'][ix]));p=torch.softmax(tensor_prior['train'][ix]+delta,dim=1)
                        prob=torch.stack([1-p[:,0],p[:,1]+p[:,3],p[:,2]+p[:,3]],dim=1)
                        loss=(((prob-y[ix])**2)*torch.tensor([.5,.25,.25],device='cuda')).sum(dim=1)
                        loss=(w[ix]*loss).mean()+penalty*delta.square().mean()
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);optimizer.step()
                    if epoch in [3,5,10,20,40]:
                        name=f'{variant}_r{penalty}_s{seed}_e{epoch}';net.eval()
                        with torch.inference_mode():p=torch.softmax(tensor_prior['validation']+2*torch.tanh(net(x['validation'])),dim=1).cpu().numpy()
                        predictions[name]=p;metrics=q.metrics(frames['validation'],p,name)
                        for row in metrics:row.update(variant=variant,penalty=penalty,seed=seed,epoch=epoch)
                        rows+=metrics;pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
                        torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},mean=mean,std=std,cols=cols,keys=keys,penalty=penalty,seed=seed,epoch=epoch),out/(name+'.pt'))
                m=next(r for r in metrics if r['dataset']=='ALL' and r['condition']=='all' and r['target']=='any')
                print(json.dumps(dict(variant=variant,penalty=penalty,seed=seed,brier=m['brier'],elapsed_s=time.perf_counter()-started)),flush=True)
    np.savez_compressed(out/'probabilities.npz',**predictions)
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,elapsed_s=time.perf_counter()-started,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interpretation='development screen; joint-state coherent residual probabilities, environment-OOF transition priors for training'),indent=2)+'\n')


if __name__=='__main__':main()
