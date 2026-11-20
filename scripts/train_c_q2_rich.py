#!/usr/bin/env python3
"""Nested train-only expert/prior ablations, then frozen validation evaluation."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.data import Dataset
from shuwei_c.motion import estimate_noise
from shuwei_c.q2_learning import load_pack,subset,score_predictions
from shuwei_c.q2_rich_learning import fit_prior,fit_gate,predict

GRID=[('basic',False,False),('basic',True,False),('rich',False,False),('rich',True,False),('rich',True,True)]
SEEDS=[20261120,20261121,20261122]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def tag(g):return f'{g[0]}_{"fitted" if g[1] else "fixed"}_{"main" if g[2] else "all"}'

def library(pack,kind,basic_count):
    if kind=='rich':return pack
    return {**pack,'experts':pack['experts'][:,:,:16],'expert_names':pack['expert_names'][:16],
            'features':pack['features'][:,:basic_count],'feature_names':pack['feature_names'][:basic_count]}

def prior_rows(meta,environments,main=False):
    mask=meta.environment.isin(environments).to_numpy()
    if main:mask&=meta.horizon_s.le(1.).to_numpy()
    priority=np.random.default_rng(20261120).random(len(meta));chosen=[]
    for _,g in meta.loc[mask].groupby(['episode','horizon_s','scenario','hand'],sort=True):
        ix=g.index.to_numpy();chosen.extend(ix[np.argsort(priority[ix],kind='stable')[:16]])
    return np.array(sorted(chosen),np.int64)

def fit_candidate(train,fit_env,held_env,noise,choice,seed,updates,basic_count,folder):
    folder.mkdir(parents=True,exist_ok=True);checkpoint=folder/f'seed{seed}.pt'
    if checkpoint.exists():
        info=json.loads(checkpoint.with_suffix('.json').read_text());assert sha(checkpoint)==info['sha256'];return torch.load(checkpoint,map_location='cpu',weights_only=False)
    fit_mask=train['meta'].environment.isin(fit_env).to_numpy()
    if choice[2]:fit_mask&=train['meta'].horizon_s.le(1.).to_numpy()
    fitting=library(subset(train,fit_mask),choice[0],basic_count)
    held=library(subset(train,train['meta'].environment.isin(held_env).to_numpy()&train['meta'].horizon_s.le(1.).to_numpy()),choice[0],basic_count) if held_env else None
    prior_path=folder/'prior.json'
    if prior_path.exists():prior=json.loads(prior_path.read_text())
    else:
        ix=prior_rows(train['meta'],fit_env,choice[2]);prior=fit_prior(library(subset(train,ix),choice[0],basic_count),noise,choice[1]);save(prior_path,prior)
        np.savez_compressed(folder/'prior_rows.npz',indices=ix)
    model=fit_gate(fitting,noise,prior,held,seed=seed,updates=updates)
    model['choice']=choice;model['fit_environments']=fit_env;model['held_environments']=held_env
    torch.save(model,checkpoint);save(checkpoint.with_suffix('.json'),dict(sha256=sha(checkpoint),choice=choice,seed=seed,fit_environments=fit_env,held_environments=held_env,
        prior_objective=prior['objective'],prior_convergence=prior['convergence'],selected_updates=model['selected_updates'],executed_updates=model['executed_updates'],
        selected_held_error=model['selected_held_error'],initial_held_error=model['initial_held_error'],fitting_rows=len(fitting['meta']),held_rows=len(held['meta']) if held else 0,
        peak_allocated_bytes=model['peak_allocated_bytes'],elapsed_s=model['elapsed_s'],trace=model['trace']))
    if held_env:
        held_indices=np.flatnonzero(train['meta'].environment.isin(held_env).to_numpy())
        held_all=library(subset(train,held_indices),choice[0],basic_count)
        predicted=predict(held_all,model)
        frame,_=score_predictions(held_all,predicted,noise)
        for k,c in enumerate('xyz'):frame['prediction_'+c]=predicted[:,k]+held_all['anchor'][:,k]
        frame['scale']=frame.dataset.map({d:n['hand_scale'] for d,n in noise.items()})
        frame.to_parquet(folder/f'seed{seed}_inner_predictions.parquet',index=False)
        np.savez_compressed(folder/f'seed{seed}_rows.npz',fitting=np.flatnonzero(fit_mask),held=held_indices)
        save(folder/f'seed{seed}_inner_predictions.json',dict(fitting_environments=fit_env,held_environments=held_env,
            noise=noise,held_main_labels_used_for_stopping=True,
            role='Inner development residuals; outer held environments absent. Not independent calibration residuals.',
            checkpoint_sha256=sha(checkpoint),rows=len(frame)))
    return model

def save_predictions(folder,pack,prediction,noise,split,fold):
    frame,summary=score_predictions(pack,prediction,noise);frame['split']=split;frame['fold']=fold;frame['method']='rich_selected'
    frame['scale']=frame.dataset.map({ds:v['hand_scale'] for ds,v in noise.items()})
    for k,c in enumerate('xyz'):frame['prediction_'+c]=prediction[:,k]+pack['anchor'][:,k]
    frame.to_parquet(folder/'ensemble_predictions.parquet',index=False);summary.to_csv(folder/'ensemble_summary.csv',index=False)

def main():
    global GRID
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('artifacts/c_q2/rich_state/data'));ap.add_argument('--out',type=Path,default=Path('artifacts/c_q2/rich_state/nested'))
    ap.add_argument('--screen-only',action='store_true');ap.add_argument('--train-only',action='store_true');ap.add_argument('--fixed-main-grid',action='store_true');ap.add_argument('--updates',type=int,default=800);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.fixed_main_grid:GRID=[('basic',False,True),('rich',False,True)]
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2);start=time.perf_counter()
    fold_path=Path('artifacts/c_q2/protocol/folds.json');folds=json.loads(fold_path.read_text())
    basic_count=len(json.loads(Path('artifacts/c_q2/learning_data/train/manifest.json').read_text())['feature_names'])
    files=[Path(__file__).resolve().relative_to(Path.cwd()),Path('src/shuwei_c/q2_rich_learning.py'),Path('src/shuwei_c/q2_rich_features.py'),Path('src/shuwei_c/q2_learning.py'),Path('src/shuwei_c/q2_features.py'),fold_path,a.data/'train/manifest.json']
    identity=dict(grid=[list(x) for x in GRID],seeds=SEEDS,updates=a.updates,prior_sample_per_episode_horizon_scenario_hand=16,prior_age_knots=[0.,1/30,.1,.2,.5,1.,2.],
        selection='Each outer fold uses only its 8 fitting / 4 inner environments. Main h<=1 inner score. Full configuration by mean inner rank, no official validation.',
        input_hashes={str(p):sha(p) for p in files},test_used=False,gpu_fraction=.2,gpu=torch.cuda.get_device_name())
    ip=a.out/'identity.json'
    if ip.exists():assert json.loads(ip.read_text())==identity
    save(ip,identity)
    for p in files:
        target=a.out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    train=load_pack(a.data/'train');records=Dataset().records('train');rankings=[]
    for fold in folds:
        number=fold['fold'];folder=a.out/f'fold{number}';folder.mkdir(exist_ok=True)
        inner_held=folds[(number+1)%4]['held_environments'];inner_fit=sorted(set(fold['training_environments'])-set(inner_held))
        assert not set(inner_fit)&set(inner_held) and not set(fold['held_environments'])&(set(inner_fit)|set(inner_held))
        noise=estimate_noise([r for r in records if r.environment in inner_fit]);assert len(noise)==3
        save(folder/'partition.json',dict(outer=fold,inner_fitting=inner_fit,inner_held=inner_held,inner_noise=noise));rows=[]
        for choice in GRID:
            model=fit_candidate(train,inner_fit,inner_held,noise,choice,SEEDS[0],a.updates,basic_count,folder/'screen'/tag(choice))
            row=dict(fold=number,tag=tag(choice),initial=model['initial_held_error'],score=model['selected_held_error'],updates=model['selected_updates'])
            rows.append(row);print(json.dumps(dict(stage='screen',**row,elapsed_s=time.perf_counter()-start)),flush=True)
        ranking=pd.DataFrame(rows);ranking['rank']=ranking.score.rank(method='min');ranking.to_csv(folder/'screen.csv',index=False);rankings.append(ranking)
        if a.screen_only:save(a.out/'fold0_screen_complete.json',dict(rows=rows,elapsed_s=time.perf_counter()-start));return
        selected=min(rows,key=lambda r:(r['score'],[tag(g) for g in GRID].index(r['tag'])))
        choice=next(g for g in GRID if tag(g)==selected['tag']);save(folder/'chosen.json',selected)
        held=library(subset(train,train['meta'].environment.isin(fold['held_environments']).to_numpy()),choice[0],basic_count);predictions=[]
        for seed in SEEDS:
            selected_model=fit_candidate(train,inner_fit,inner_held,noise,choice,seed,a.updates,basic_count,folder/'screen'/tag(choice))
            model=fit_candidate(train,fold['training_environments'],[],fold['noise'],choice,seed,selected_model['selected_updates'],basic_count,folder/'outer')
            predictions.append(predict(held,model))
        save_predictions(folder,held,np.mean(predictions,axis=0),fold['noise'],'train_oof',number)
        print(json.dumps(dict(stage='outer_complete',fold=number,choice=tag(choice),elapsed_s=time.perf_counter()-start)),flush=True)
    rankings=pd.concat(rankings);rankings.to_csv(a.out/'inner_screens.csv',index=False)
    chosen=rankings.groupby('tag')['rank'].mean().sort_values(kind='stable').index[0];choice=next(g for g in GRID if tag(g)==chosen)
    root=a.out/'full';root.mkdir(exist_ok=True);full_env=sorted(train['meta'].environment.unique());noise=estimate_noise(records)
    inner_held=folds[0]['held_environments'];inner_fit=folds[0]['training_environments'];inner_noise=folds[0]['noise'];models=[]
    for seed in SEEDS:
        selected=fit_candidate(train,inner_fit,inner_held,inner_noise,choice,seed,a.updates,basic_count,root/'selection')
        models.append(fit_candidate(train,full_env,[],noise,choice,seed,selected['selected_updates'],basic_count,root))
    save(root/'freeze.json',dict(choice=choice,checkpoint_hashes={str(root/f'seed{s}.pt'):sha(root/f'seed{s}.pt') for s in SEEDS},official_validation_not_read=True,elapsed_s=time.perf_counter()-start))
    if a.train_only:print('Training and full model freeze complete; official validation not read',flush=True);return
    validation=library(load_pack(a.data/'validation'),choice[0],basic_count)
    save_predictions(root,validation,np.mean([predict(validation,m) for m in models],axis=0),noise,'validation',-1)
    save(a.out/'manifest.json',dict(status='PASS',chosen=chosen,elapsed_s=time.perf_counter()-start,test_used=False,full_freeze_sha256=sha(root/'freeze.json')))

if __name__=='__main__':main()
