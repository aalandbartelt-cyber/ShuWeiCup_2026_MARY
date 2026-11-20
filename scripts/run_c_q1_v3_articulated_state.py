#!/usr/bin/env python3
"""Articulation and stereo-residual ablations on the fixed numerical predictor."""
from pathlib import Path
import sys,json,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from shuwei_c.q1v3_learning import predict_gate,network
from shuwei_c.q1v3_state import row_weights,score
from run_c_q1_v3_articulated_risk import select_visual


def main():
    source=Path('artifacts/c_q1_v3/articulated_features');assert (source/'manifest.json').exists();out=Path('artifacts/c_q1_v3/articulated_state');out.mkdir(exist_ok=True);start=time.perf_counter();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    packs={};visual={};base={};seeds=[20261120,20261121,20261122]
    for split in ['train','validation']:
        p=dict(np.load(f'artifacts/c_q1_v3/state_screen/{split}_0.1.npz'));p['meta']=pd.read_parquet(f'artifacts/c_q1_v3/state_screen/{split}_0.1_meta.parquet');packs[split]=p;visual[split]=dict(np.load(source/f'state_{split}.npz'))
        predictions=[]
        for seed in seeds:
            m=torch.load(f'artifacts/c_q1_v3/state_screen/gate_r0.001_seed{seed}_e10_0.1.pt',map_location='cpu',weights_only=False);predictions.append(predict_gate(p,m))
        base[split]=np.mean(predictions,axis=0)
    w=row_weights(packs['train']['meta']);rows=[];parameters=[];predictions={'numeric_frozen':base['validation']};names=json.loads(Path('artifacts/c_q1_v3/state_screen/feature_names.json').read_text());flag_column=names.index('flag_now')
    def record(method,p):
        meta=packs['validation']['meta'];error=np.linalg.norm(p-packs['validation']['target'],axis=1)
        for ds in ['ALL','EgoDemo','EgoPro','EgoStandard']:
            for condition in ['all','current_flagged']:
                mask=np.ones(len(meta),bool) if ds=='ALL' else meta.dataset.to_numpy()==ds
                if condition!='all':mask &=packs['validation']['features'][:,flag_column]>0
                if mask.any():rows.append(dict(method=method,dataset=ds,condition=condition,error=score(meta.loc[mask],error[mask]),n=int(mask.sum())))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    record('numeric_frozen',base['validation'])
    for condition in ['all','current_flagged']:
        corrections={s:visual[s]['anchor_correction'].copy() for s in packs}
        if condition!='all':
            for s in packs:corrections[s]*=(packs[s]['features'][:,flag_column]>0)[:,None]
        loss=lambda alpha:float(np.dot(w,np.linalg.norm(base['train']+alpha*corrections['train']-packs['train']['target'],axis=1)))
        result=minimize_scalar(loss,bounds=(0,.5),method='bounded');alpha=float(result.x);name='stereo_'+condition
        p=base['validation']+alpha*corrections['validation'];predictions[name]=p;record(name,p);parameters.append(dict(method=name,alpha=alpha,train_objective=float(result.fun)))
    weight=torch.tensor(w*len(w),device='cuda',dtype=torch.float32);target=torch.tensor(packs['train']['target'],device='cuda');prior={s:torch.tensor(p,device='cuda') for s,p in base.items()}
    experts={s:torch.tensor(np.concatenate([packs[s]['experts'],visual[s]['anchor_correction'][:,:,None]],axis=-1),device='cuda',dtype=torch.float32) for s in packs}
    for kind in ['numeric','shape','motion']:
        raw={s:np.concatenate([packs[s]['features'],*([] if kind=='numeric' else [select_visual(visual[s]['features'][:,None,:],kind)[:,0],visual[s]['anchor_quality']])],axis=-1) for s in packs}
        mean=np.average(raw['train'],weights=w,axis=0);std=np.maximum(np.sqrt(np.average((raw['train']-mean)**2,weights=w,axis=0)),1e-4);x={s:torch.tensor(np.clip((a-mean)/std,-4,4),device='cuda',dtype=torch.float32) for s,a in raw.items()}
        for reg in [.01,.1]:
            for seed in seeds:
                torch.manual_seed(seed);net=torch.nn.Sequential(torch.nn.Linear(x['train'].shape[1],32),torch.nn.SiLU(),torch.nn.Dropout(.3),torch.nn.Linear(32,experts['train'].shape[-1])).cuda();torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias);optimizer=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.1)
                for epoch in range(1,21):
                    net.train()
                    for ix in torch.randperm(len(w),device='cuda').split(4096):
                        delta=.15*torch.tanh(net(x['train'][ix]))
                        # Pure-numeric ablation cannot access the visual anchor.
                        if kind=='numeric':delta=torch.cat([delta[:,:-1],torch.zeros_like(delta[:,-1:])],dim=1)
                        p=prior['train'][ix]+torch.einsum('nck,nk->nc',experts['train'][ix],delta)
                        loss=(torch.sqrt(((p-target[ix])**2).sum(dim=1)+1e-8)*weight[ix]).mean()+reg*delta.square().mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);optimizer.step()
                    if epoch in [5,10,20]:
                        name=f'{kind}_r{reg}_s{seed}_e{epoch}';net.eval()
                        with torch.inference_mode():
                            delta=.15*torch.tanh(net(x['validation']))
                            if kind=='numeric':delta[:,-1]=0
                            p=(prior['validation']+torch.einsum('nck,nk->nc',experts['validation'],delta)).cpu().numpy()
                        predictions[name]=p;record(name,p);torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},mean=mean,std=std,kind=kind,regularization=reg,seed=seed,epoch=epoch),out/(name+'.pt'))
                print(json.dumps(dict(variant=kind,regularization=reg,seed=seed,error=rows[-8]['error'] if len(rows)>=8 else rows[0]['error'],seconds=time.perf_counter()-start)),flush=True)
    np.savez_compressed(out/'predictions.npz',**predictions);(out/'stereo_weights.json').write_text(json.dumps(parameters,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,elapsed_s=time.perf_counter()-start,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='Development screening with frozen full-training numerical predictor; conditional flagged subgroup may be small; confirm any selected visual increment across held environments'),indent=2)+'\n')


if __name__=='__main__':main()
