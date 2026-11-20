#!/usr/bin/env python3
"""Frozen numerical gate plus learned visual residual: fixed validation screening.

AI-assisted by Codex, 2026-11-20. No official test inputs; report every candidate.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
import torch
from shuwei_c.q1v3_visual import fit_pca,align_visual
from shuwei_c.q1v3_learning import network,base_coefficients
from shuwei_c.q1v3_state import row_weights,score


def load_pack(split,horizon=.1):
    if horizon==.1:
        source=Path('artifacts/c_q1_v3/state_screen');a=dict(np.load(source/f'{split}_0.1.npz'))
        a['meta']=pd.read_parquet(source/f'{split}_0.1_meta.parquet');return a
    from shuwei_c.data import Dataset
    from shuwei_c.q1v3_state import pack_records
    noise=json.loads(Path('artifacts/c_q1_v3/horizon_transfer/protocol.json').read_text())['noise']
    return pack_records(Dataset().records(split),horizon,json.loads(Path('configs/c_q1_v2.json').read_text()),noise)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--oof-numeric',action='store_true');parser.add_argument('--horizon',type=float,choices=[.1,.5],default=.1);args=parser.parse_args()
    if args.oof_numeric and args.horizon!=.1:raise ValueError('OOF numeric checkpoints currently exist only for 0.1 seconds')
    root=Path('artifacts/c_q1_v3/visual_dino15');cache=Path('artifacts/c_q1_v3/visual_screen')
    out=Path('artifacts/c_q1_v3/visual_screen_oof') if args.oof_numeric else cache
    if args.horizon!=.1:cache=out=Path(f'artifacts/c_q1_v3/visual_screen_h{args.horizon}')
    out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Completed screen exists; do not overwrite')
    if not (root/'manifest.json').exists():raise RuntimeError('Complete 166-video manifest required')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25);started=time.perf_counter()
    train=load_pack('train',args.horizon);val=load_pack('validation',args.horizon)
    if args.horizon!=.1:
        visual={}
        for split,pack in [('train',train),('validation',val)]:
            frame=pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_{split}_0.1.parquet');meta=pack['meta']
            ix=pd.MultiIndex.from_frame(frame[['episode','origin']]).get_indexer(pd.MultiIndex.from_frame(meta[['episode','origin']]))
            assert (ix>=0).all();np.testing.assert_array_equal(frame.time_s.to_numpy()[ix],meta.origin_s.to_numpy())
            a=np.load(f'artifacts/c_q1_v3/video_quality/visual_{split}.npz');visual[split]={key:a[key].reshape(len(frame),2,-1)[ix,meta.hand.to_numpy()] for key in a.files}
            np.savez_compressed(out/f'{split}_pack.npz',features=pack['features'],experts=pack['experts'],target=pack['target']);meta.to_parquet(out/f'{split}_meta.parquet',index=False)
    elif (cache/'visual_train.npz').exists() and (cache/'visual_validation.npz').exists():
        visual={split:dict(np.load(cache/f'visual_{split}.npz')) for split in ['train','validation']}
    else:
        pca=fit_pca(root,set(train['meta'].episode));joblib.dump(pca,out/'pca_train_only.joblib');visual={}
        for split,pack in [('train',train),('validation',val)]:
            visual[split],names=align_visual(pack['meta'],root,pca);np.savez_compressed(out/f'visual_{split}.npz',**visual[split])
        (out/'visual_feature_names.json').write_text(json.dumps(names,indent=2)+'\n')
    # Fixed ensemble point prediction; each component's coefficient is known.
    tensors={};seeds=[20261120,20261121,20261122]
    for split,pack in [('train',train),('validation',val)]:
        coeff=[]
        for seed in seeds:
            path=f'artifacts/c_q1_v3/state_screen/gate_r0.001_seed{seed}_e10_0.1.pt' if args.horizon==.1 else f'artifacts/c_q1_v3/horizon_transfer/gate_h{args.horizon:.6f}_s{seed}.pt'
            m=torch.load(path,map_location='cpu',weights_only=False)
            x=torch.tensor(np.clip((pack['features']-m['mean'])/m['std'],-10,10),device='cuda',dtype=torch.float32)
            net=network(x.shape[1],pack['experts'].shape[-1]).cuda();net.load_state_dict(m['state_dict']);net.eval()
            with torch.inference_mode():c=np.concatenate([(.5*torch.tanh(net(b))).cpu().numpy() for b in x.split(4096)])
            coeff.append(base_coefficients(pack['meta'],m['base'])+c)
        tensors[split]=dict(e=torch.tensor(pack['experts'],device='cuda'),y=torch.tensor(pack['target'],device='cuda'),b=torch.tensor(np.mean(coeff,axis=0),device='cuda'))
    if args.oof_numeric:
        frame=pd.read_parquet('artifacts/c_q1_v3/numeric_oof/train_oof.parquet')
        keys=['episode','origin','hand'];index=train['meta'][keys].reset_index(names='pack_index').merge(frame[keys+['fold']],on=keys,validate='one_to_one')
        assembled=np.zeros_like(tensors['train']['b'].cpu().numpy())
        for fold in sorted(frame.fold.unique()):
            g=frame[frame.fold==fold].copy();a=g[keys].merge(index,on=keys,validate='one_to_one');coeff=[]
            for seed in seeds:
                m=torch.load(f'artifacts/c_q1_v3/numeric_oof/fold{fold}_seed{seed}.pt',map_location='cpu',weights_only=False)
                features=g[['x_'+k for k in m['feature_names']]].to_numpy()
                x=torch.tensor(np.clip((features-m['mean'])/m['std'],-10,10),device='cuda',dtype=torch.float32)
                net=network(x.shape[1],train['experts'].shape[-1]).cuda();net.load_state_dict(m['state_dict']);net.eval()
                with torch.inference_mode():c=np.concatenate([(.5*torch.tanh(net(b))).cpu().numpy() for b in x.split(4096)])
                coeff.append(base_coefficients(g,m['base'])+c)
            assembled[a.pack_index.to_numpy()]=np.mean(coeff,axis=0)
        assert len(index)==len(train['meta'])
        tensors['train']['b']=torch.tensor(assembled,device='cuda')
    w=row_weights(train['meta']);weight=torch.tensor(w*len(w),device='cuda',dtype=torch.float32)
    rows=[];predictions={};variants={'numeric_refit':[], 'diagnostics':['diagnostics'],'appearance':['appearance'],'visual_motion':['diagnostics','motion'],'all_visual':['diagnostics','appearance','motion']}
    with torch.inference_mode():baseline=torch.einsum('nck,nk->nc',tensors['validation']['e'],tensors['validation']['b']).cpu().numpy()
    predictions['numeric_frozen']=baseline
    def record(name,pred,**extra):
        err=np.linalg.norm(pred-val['target'],axis=1)
        for ds in ['ALL','EgoDemo','EgoPro','EgoStandard']:
            mask=np.ones(len(err),bool) if ds=='ALL' else val['meta'].dataset.to_numpy()==ds
            rows.append(dict(method=name,dataset=ds,score=score(val['meta'].loc[mask],err[mask]),**extra))
        pd.DataFrame(rows).to_csv(out/'scores.csv',index=False)
    record('numeric_frozen',baseline)
    for variant,keys in variants.items():
        feature={split:np.concatenate([pack['features'],*[visual[split][k] for k in keys]],axis=1) for split,pack in [('train',train),('validation',val)]}
        mean=np.average(feature['train'],weights=w,axis=0);std=np.maximum(np.sqrt(np.average((feature['train']-mean)**2,weights=w,axis=0)),1e-4)
        tx={split:torch.tensor(np.clip((feature[split]-mean)/std,-8,8),device='cuda',dtype=torch.float32) for split in feature}
        for reg in [.01,.1]:
            for seed in seeds:
                torch.manual_seed(seed);net=network(tx['train'].shape[1],train['experts'].shape[-1]).cuda();torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
                optimizer=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.05);a=tensors['train'];v=tensors['validation']
                for epoch in range(1,21):
                    net.train()
                    for ix in torch.randperm(len(w),device='cuda').split(4096):
                        delta=.25*torch.tanh(net(tx['train'][ix]));pred=torch.einsum('nck,nk->nc',a['e'][ix],a['b'][ix]+delta)
                        loss=(torch.sqrt(((pred-a['y'][ix])**2).sum(dim=1)+1e-8)*weight[ix]).mean()+reg*delta.square().mean()
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);optimizer.step()
                    if epoch in [3,5,10,20]:
                        name=f'{variant}_r{reg}_s{seed}_e{epoch}';net.eval()
                        with torch.inference_mode():
                            pred=np.concatenate([torch.einsum('nck,nk->nc',v['e'][sl],v['b'][sl]+.25*torch.tanh(net(tx['validation'][sl]))).cpu().numpy() for sl in [slice(i,i+4096) for i in range(0,len(val['meta']),4096)]])
                        predictions[name]=pred;record(name,pred,variant=variant,reg=reg,seed=seed,epoch=epoch)
                        torch.save(dict(state_dict=net.cpu().state_dict(),mean=mean,std=std,keys=keys,reg=reg,seed=seed,epoch=epoch),out/(name+'.pt'));net.cuda()
                print(json.dumps(dict(variant=variant,reg=reg,seed=seed,score=rows[-4]['score'],elapsed_s=time.perf_counter()-started)),flush=True)
    np.savez_compressed(out/'predictions.npz',**predictions)
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,horizon=args.horizon,numeric_training_oof=args.oof_numeric,elapsed_s=time.perf_counter()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interpretation='development screening; fixed numeric ensemble plus residual; visual PCA fitted on train videos only'),indent=2)+'\n')


if __name__=='__main__':main()
