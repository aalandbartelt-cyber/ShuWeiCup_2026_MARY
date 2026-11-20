#!/usr/bin/env python3
"""Environment-held-out verification of the promising numeric gate.

AI-assisted by Codex, 2026-11-20. Parameters selected in development screening;
no official test data used. Baselines refit on exactly the same fold records.
"""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from shuwei_c.data import Dataset
from shuwei_c.motion import estimate_noise
from shuwei_c.q1v2_state import linear_moments,fit_linear
from shuwei_c.q1v3_state import pack_records,fit_distance,score
from shuwei_c.q1v3_learning import fit_gate,predict_gate


def attach(pack,predictions,fold):
    f=pack['meta'].copy();f['fold']=fold
    for k,name in enumerate(pack['feature_names']):f['x_'+name]=pack['features'][:,k]
    for name,p in predictions.items():
        error=np.linalg.norm(p-pack['target'],axis=1);f[name+'_normalized_error']=error;f[name+'_error']=error*f.scale
    return f


def main():
    out=Path('artifacts/c_q1_v3/numeric_oof');out.mkdir(exist_ok=True);screen=Path('artifacts/c_q1_v3/state_screen');data=Dataset();records=data.records('train');config=json.loads(Path('configs/c_q1_v2.json').read_text())
    ensembles=pd.read_csv(screen/'ensemble_comparison.csv');selected=ensembles.loc[ensembles.score.idxmin()];epoch=int(selected.epochs);reg=float(selected.regularization)
    seeds=[20261120,20261121,20261122];checkpoint=torch.load(screen/f'gate_r{reg}_seed{seeds[0]}_e{epoch}_0.1.pt',map_location='cpu',weights_only=False);base_config=checkpoint['base']
    protocol=dict(horizon=.1,epochs=epoch,regularization=reg,seeds=seeds,base_config=base_config,folds=4,
        status='development-selected hyperparameters; whole-environment held-out fitting; official test unused')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');torch.cuda.set_per_process_memory_fraction(.16);frames=[];folds=[];started=time.perf_counter()
    for fold,(fi,hi) in enumerate(GroupKFold(4).split(records,groups=[r.environment for r in records])):
        fitting=[records[i] for i in fi];held=[records[i] for i in hi];assert not ({r.environment for r in fitting}&{r.environment for r in held})
        noise=estimate_noise(fitting);train=pack_records(fitting,.1,config,noise);validation=pack_records(held,.1,config,noise)
        base=fit_distance(train,base_config['penalty'],base_config['dataset_specific']);pred=[]
        for seed in seeds:
            model=fit_gate(train,base,epoch,reg,seed);torch.save(model,out/f'fold{fold}_seed{seed}.pt');pred.append(predict_gate(validation,model))
        moments=linear_moments(fitting,.1,config,noise);scalar=fit_linear(moments,'scalar_damped');multi=fit_linear(moments,'multiscale',1.)
        e=validation['experts'];prediction=dict(gate=np.mean(pred,axis=0),linear=e[...,0],scalar=e[...,0]*scalar[0],multiscale=np.einsum('nck,k->nc',e[...,[0,2,4,6]],multi))
        f=attach(validation,prediction,fold);frames.append(f);pd.concat(frames,ignore_index=True).to_parquet(out/'train_oof.parquet',index=False)
        result={name:score(validation['meta'],f[name+'_normalized_error']) for name in prediction}
        folds.append(dict(fold=fold,training_environments=sorted({r.environment for r in fitting}),held_environments=sorted({r.environment for r in held}),noise=noise,base=base,scores=result))
        print(json.dumps(dict(fold=fold,**result)),flush=True)
    # Reuse the actual full-training checkpoints selected above, not a test fit.
    noise=estimate_noise(records);validation=pack_records(data.records('validation'),.1,config,noise);pred=[]
    for seed in seeds:
        model=torch.load(screen/f'gate_r{reg}_seed{seed}_e{epoch}_0.1.pt',map_location='cpu',weights_only=False);pred.append(predict_gate(validation,model))
    v2=json.loads(Path('artifacts/c_q1_v2/state_models_development.json').read_text())['horizons']['0.1']['family_models'];e=validation['experts']
    predictions=dict(gate=np.mean(pred,axis=0),linear=e[...,0],scalar=e[...,0]*v2['scalar_damped']['coefficients'][0],multiscale=np.einsum('nck,k->nc',e[...,[0,2,4,6]],v2['multiscale']['coefficients']))
    f=attach(validation,predictions,-1);f.to_parquet(out/'validation.parquet',index=False)
    rows=[]
    for split,frame in [('train_oof',pd.concat(frames,ignore_index=True)),('validation',f)]:
        for ds in ['ALL','EgoDemo','EgoPro','EgoStandard']:
            g=frame if ds=='ALL' else frame[frame.dataset==ds]
            for name in predictions:rows.append(dict(split=split,dataset=ds,method=name,score=score(g,g[name+'_normalized_error']),mean_error=score(g,g[name+'_error']),environments=g.environment.nunique(),n=len(g)))
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False);(out/'folds.json').write_text(json.dumps(folds,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(elapsed_s=time.perf_counter()-started,test_used=False,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/q1v3_state.py'),Path('src/shuwei_c/q1v3_learning.py')]}),indent=2)+'\n');print('Numeric OOF complete',flush=True)


if __name__=='__main__':main()
