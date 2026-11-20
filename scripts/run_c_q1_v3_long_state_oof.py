#!/usr/bin/env python3
"""0.5-second video increment with nested numerical OOF residual training."""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from shuwei_c.data import Dataset
from shuwei_c.motion import estimate_noise
from shuwei_c.q1v3_state import pack_records,fit_distance,score
from shuwei_c.q1v3_learning import fit_gate,predict_gate
from shuwei_c import q1v3_residual as residual


def align(source,target,values):
    keys=['episode','origin','hand'];ix=pd.MultiIndex.from_frame(source[keys]).get_indexer(pd.MultiIndex.from_frame(target[keys]));assert (ix>=0).all()
    return values[ix]


def main():
    out=Path('artifacts/c_q1_v3/long_state_oof');out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Completed run exists')
    start=time.perf_counter();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
    data=Dataset();records=data.records('train');config=json.loads(Path('configs/c_q1_v2.json').read_text());seeds=[20261120,20261121,20261122];visual={};qframes={}
    for split in ['train','validation']:
        qframes[split]=pd.read_parquet(f'artifacts/c_q1_v2/quality_numerical/features_{split}_0.1.parquet')
        visual[split]=np.load(f'artifacts/c_q1_v3/video_quality/visual_{split}.npz')['diagnostics'].reshape(len(qframes[split]),2,-1)
    def features(pack,split,video):
        if not video:return pack['features']
        f=pack['meta'];ix=pd.MultiIndex.from_frame(qframes[split][['episode','origin']]).get_indexer(pd.MultiIndex.from_frame(f[['episode','origin']]))
        assert (ix>=0).all();return np.c_[pack['features'],visual[split][ix,f.hand.to_numpy()]]
    def fitted(fr,hr,tag):
        noise=estimate_noise(fr);a=pack_records(fr,.5,config,noise);b=pack_records(hr,.5,config,noise);base=fit_distance(a,.01,True);pred=[]
        for seed in seeds:
            m=fit_gate(a,base,10,.001,seed);torch.save(m,out/f'{tag}_s{seed}.pt');pred.append(predict_gate(b,m))
        return a,b,np.mean(pred,axis=0)
    parts=[];oof_vectors=[];folds=[]
    for fold,(fi,hi) in enumerate(GroupKFold(4).split(records,groups=[r.environment for r in records])):
        fr=[records[i] for i in fi];hr=[records[i] for i in hi];a,b,bp=fitted(fr,hr,f'fold{fold}_base');innerframes=[];innervalues=[]
        for inner,(ii,jj) in enumerate(GroupKFold(3).split(fr,groups=[r.environment for r in fr])):
            _,held,pred=fitted([fr[i] for i in ii],[fr[i] for i in jj],f'fold{fold}_inner{inner}')
            innerframes.append(held['meta']);innervalues.append(pred*held['meta'].scale.to_numpy()[:,None])
        ap=align(pd.concat(innerframes,ignore_index=True),a['meta'],np.concatenate(innervalues))/a['meta'].scale.to_numpy()[:,None]
        predictions={'gate':bp}
        for video in [False,True]:
            name='video' if video else 'numeric_refit';raw=features(a,'train',video);vr=features(b,'train',video);values=[]
            for seed in seeds:
                m=residual.fit(a,raw,ap,seed);torch.save(m,out/f'fold{fold}_{name}_s{seed}.pt');values.append(residual.predict(b,vr,bp,m))
            predictions[name]=np.mean(values,axis=0)
        f=b['meta'].copy();f['fold']=fold
        for name,p in predictions.items():f[name+'_error']=np.linalg.norm(p-b['target'],axis=1)*f.scale
        parts.append(f);oof_vectors.append(bp*f.scale.to_numpy()[:,None]);scores={name:score(f,f[name+'_error']) for name in predictions}
        folds.append(dict(fold=fold,held_environments=sorted(f.environment.unique()),scores=scores));print(json.dumps(dict(folds[-1],elapsed_s=time.perf_counter()-start)),flush=True)
    oof=pd.concat(parts,ignore_index=True);oof.to_parquet(out/'train_oof.parquet',index=False)
    noise=estimate_noise(records);a=pack_records(records,.5,config,noise);b=pack_records(data.records('validation'),.5,config,noise)
    ap=align(oof,a['meta'],np.concatenate(oof_vectors))/a['meta'].scale.to_numpy()[:,None];bp=np.mean([predict_gate(b,torch.load(f'artifacts/c_q1_v3/horizon_transfer/gate_h0.500000_s{s}.pt',map_location='cpu',weights_only=False)) for s in seeds],axis=0)
    predictions={'gate':bp}
    for video in [False,True]:
        name='video' if video else 'numeric_refit';raw=features(a,'train',video);vr=features(b,'validation',video);values=[]
        for seed in seeds:
            m=residual.fit(a,raw,ap,seed);torch.save(m,out/f'full_{name}_s{seed}.pt');values.append(residual.predict(b,vr,bp,m))
        predictions[name]=np.mean(values,axis=0)
    val=b['meta'].copy()
    for name,p in predictions.items():val[name+'_error']=np.linalg.norm(p-b['target'],axis=1)*val.scale
    val.to_parquet(out/'validation.parquet',index=False);np.savez_compressed(out/'validation_predictions.npz',**predictions)
    rows=[]
    for split,f in [('train_oof',oof),('validation',val)]:
        for ds in ['ALL','EgoDemo','EgoPro','EgoStandard']:
            g=f if ds=='ALL' else f[f.dataset==ds]
            for name in predictions:rows.append(dict(split=split,dataset=ds,method=name,native_error=score(g,g[name+'_error']),normalized_error=score(g,g[name+'_error']/g.scale)))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);(out/'folds.json').write_text(json.dumps(folds,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,horizon=.5,elapsed_s=time.perf_counter()-start,scope='Outer held environments excluded from numerical gate, inner residual targets and visual residual; pure diagnostics require no PCA; candidate selected in development',source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_residual.py')]}),indent=2)+'\n')


if __name__=='__main__':main()
