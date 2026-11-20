#!/usr/bin/env python3
"""Strict nested-environment static-gate study with update-based selection.

The candidate family is fixed in advance. Every outer fold selects sampling,
batch size and training duration using only its other twelve environments.
No official test record is loaded. Official validation is evaluated only after
all training-only configuration choices have been saved.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.data import Dataset
from shuwei_c.motion import estimate_noise
from shuwei_c.q2_learning import load_pack, subset, normalized, score_predictions
from shuwei_c.q2_sequence import SequenceExpertGate

SEEDS = (20261120, 20261121, 20261122)
GRID = [(p, b) for p in (4, 16, 0) for b in (256, 1024)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, content):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(content, indent=2)+'\n')
    temporary.replace(path)


def choose_rows(meta, environments, per_group, sample_seed=20261120):
    mask = meta.environment.isin(environments).to_numpy()
    if per_group == 0:
        return np.flatnonzero(mask)
    # A fixed random rank on the global row order makes p=4 a subset of p=16.
    priority = np.random.default_rng(sample_seed).random(len(meta))
    chosen = []
    for _, group in meta.loc[mask].groupby(['episode', 'horizon_s', 'scenario', 'hand'], sort=True):
        index = group.index.to_numpy()
        chosen.extend(index[np.argsort(priority[index], kind='stable')[:per_group]])
    return np.array(sorted(chosen), np.int64)


def tensors(pack, noise, mean, std, device='cuda'):
    raw = normalized(pack, noise)
    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)
    return dict(x=tensor(np.clip((raw['x']-mean)/std, -10, 10)),
                e=tensor(raw['e']), y=tensor(raw['y']),
                w=tensor(raw['weights']*len(raw['weights'])))


def forward(model, x, experts):
    return torch.einsum('nck,nk->nc', experts, torch.softmax(model.gate(x), dim=-1))


def held_loss(model, data, batch=16384):
    model.eval(); total = 0.
    with torch.inference_mode():
        for begin in range(0, len(data['x']), batch):
            take = slice(begin, begin+batch)
            p = forward(model, data['x'][take], data['e'][take])
            total += float((torch.linalg.vector_norm(p-data['y'][take], dim=1)*data['w'][take]).sum())
    return total/len(data['x'])


def fit(train, noise, seed, batch, updates, held=None, every=25, patience=8):
    torch.manual_seed(seed)
    raw = normalized(train, noise); weights = raw['weights']
    mean = np.average(raw['x'], axis=0, weights=weights)
    std = np.maximum(np.sqrt(np.average((raw['x']-mean)**2, axis=0, weights=weights)), 1e-4)
    data = tensors(train, noise, mean, std)
    validation = tensors(held, noise, mean, std) if held is not None else None
    model = SequenceExpertGate(len(mean), hidden=32, sequence=False).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator(device='cuda'); generator.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    best = held_loss(model, validation) if validation is not None else float('inf')
    initial = best if validation is not None else None
    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    best_update, stale, trace, count, epochs, seen = 0, 0, [], 0, 0, 0
    rolling, rolling_count = 0., 0
    if held is not None:
        trace.append(dict(optimizer_updates=0, train_objective=None, inner_held_error=initial, examples_seen=0, epochs_started=0, elapsed_s=0.))
    while count < updates:
        epochs += 1
        for ix in torch.randperm(len(data['x']), generator=generator, device='cuda').split(batch):
            model.train()
            p = forward(model, data['x'][ix], data['e'][ix])
            loss = (torch.sqrt((p-data['y'][ix]).square().sum(1)+1e-8)*data['w'][ix]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite static-gate objective')
            optimizer.zero_grad(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite static-gate gradient')
            optimizer.step(); count += 1; seen += len(ix)
            rolling += float(loss.detach()); rolling_count += 1
            if count % every == 0 or count == updates:
                score = held_loss(model, validation) if validation is not None else None
                trace.append(dict(optimizer_updates=count, train_objective=rolling/rolling_count,
                                  inner_held_error=score, examples_seen=seen, epochs_started=epochs,
                                  elapsed_s=time.perf_counter()-started))
                rolling, rolling_count = 0., 0
                if held is None or score < best-1e-6:
                    if held is not None:
                        best = score
                    best_update, stale = count, 0
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                elif held is not None:
                    stale += 1
            if count >= updates or (held is not None and stale >= patience):
                break
        if held is not None and stale >= patience:
            break
    record = dict(state_dict=best_state, kind='static_gate', hidden=32, observed_mean=np.zeros(9),
                  feature_mean=mean, feature_std=std, noise=noise, feature_names=train['feature_names'],
                  expert_names=train['expert_names'], seed=seed, batch_size=batch,
                  selected_updates=best_update, executed_updates=count, examples_seen=seen,
                  initial_inner_error=initial, selected_inner_error=best if held is not None else None,
                  trace=trace, elapsed_s=time.perf_counter()-started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  optimizer='torch.optim.AdamW', lr=.001, weight_decay=.01, gradient_clip=2.)
    del data, validation, model, optimizer, raw
    torch.cuda.empty_cache()
    return record


def predict(pack, model, batch=16384):
    raw = normalized(pack, model['noise'])
    network = SequenceExpertGate(len(model['feature_names']), hidden=32, sequence=False).cuda()
    network.load_state_dict(model['state_dict']); network.eval()
    result = []
    with torch.inference_mode():
        for begin in range(0, len(raw['x']), batch):
            take = slice(begin, begin+batch)
            x = torch.as_tensor(np.clip((raw['x'][take]-model['feature_mean'])/model['feature_std'], -10, 10), device='cuda', dtype=torch.float32)
            experts = torch.as_tensor(raw['e'][take], device='cuda', dtype=torch.float32)
            result.append(forward(network, x, experts).cpu().numpy()*raw['scales'][take, None])
    del network
    torch.cuda.empty_cache()
    return np.concatenate(result)


def save_model(path, model, metadata):
    temporary = path.with_suffix('.tmp.pt'); torch.save(model, temporary); temporary.replace(path)
    selection = {k:v for k,v in model.items() if k not in ('state_dict','feature_mean','feature_std','observed_mean')}
    selection.update(metadata)
    selection['checkpoint_sha256'] = sha(path)
    save_json(path.with_suffix('.json'), selection)
    pd.DataFrame(model['trace']).to_csv(path.with_name(path.stem+'_trace.csv'), index=False)


def load_completed(path):
    if not path.exists() or not path.with_suffix('.json').exists():
        return None
    record = json.loads(path.with_suffix('.json').read_text())
    if record['checkpoint_sha256'] != sha(path):
        raise ValueError('Static gate checkpoint hash differs')
    return torch.load(path, map_location='cpu', weights_only=False)


def save_predictions(folder, pack, displacement, noise, seed, fold, split):
    frame, summary = score_predictions(pack, displacement, noise)
    frame['fold'], frame['seed'], frame['split'] = fold, seed, split
    frame['scale'] = frame.dataset.map({k:v['hand_scale'] for k,v in noise.items()})
    frame['method'] = 'static_nested'
    for k,c in enumerate('xyz'):
        frame['prediction_'+c] = displacement[:,k]+pack['anchor'][:,k]
    suffix = 'ensemble' if seed == -1 else f'seed{seed}'
    target=folder/f'{suffix}_predictions.parquet';temp=target.with_suffix('.tmp.parquet')
    frame.to_parquet(temp,index=False);temp.replace(target)
    summary['fold'],summary['seed'],summary['split'] = fold,seed,split
    summary['method']='static_nested'
    summary.to_csv(folder/f'{suffix}_summary.csv',index=False)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('artifacts/c_q2/learning_data'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/c_q2/static_gate/nested'))
    parser.add_argument('--screen-only',action='store_true')
    parser.add_argument('--updates',type=int,default=1200)
    parser.add_argument('--every',type=int,default=25)
    parser.add_argument('--patience',type=int,default=8)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    folds_path=Path('artifacts/c_q2/protocol/folds.json');folds=json.loads(folds_path.read_text())
    files=[Path(__file__),Path('src/shuwei_c/q2_sequence.py'),Path('src/shuwei_c/q2_learning.py'),
           Path('src/shuwei_c/q2_features.py'),Path('src/shuwei_c/motion.py'),Path('src/shuwei_c/data.py'),
           folds_path,args.data/'train/manifest.json',args.data/'validation/manifest.json']
    identity=dict(grid=[dict(per_group=p,batch=b) for p,b in GRID],seeds=SEEDS,max_updates=args.updates,
                  eval_every_updates=args.every,patience_checks=args.patience,selection_horizon_max_s=1.,
                  fitting_includes_2s=True,input_hashes={str(p):sha(p) for p in files},
                  selection='strict outer environments excluded from all normalization, screening and stopping',
                  full_configuration_rule='minimum mean inner rank across 4 outer screens; ties more data then smaller batch',
                  test_used=False,hardware=torch.cuda.get_device_name(),torch_version=torch.__version__,gpu_fraction=.2)
    path=args.out/'identity.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(identity)):
        raise ValueError('Inputs or implementation changed; choose a new static-gate directory')
    save_json(path,identity)
    for p in files:
        relative=p.relative_to(Path.cwd()) if p.is_absolute() else p
        dst=args.out/'source_snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    train=load_pack(args.data/'train');records=Dataset().records('train')
    all_screen=[];summaries=[];started=time.perf_counter()
    for outer in folds:
        number=outer['fold'];folder=args.out/f'fold{number}';folder.mkdir(exist_ok=True)
        inner_held=folds[(number+1)%4]['held_environments']
        inner_fit=sorted(set(outer['training_environments'])-set(inner_held))
        assert not(set(inner_fit)&set(inner_held) or set(outer['held_environments'])&(set(inner_fit)|set(inner_held)))
        inner_noise=estimate_noise([r for r in records if r.environment in inner_fit])
        if len(inner_noise)!=3:raise ValueError('Inner fitting loses a dataset')
        held_ix=np.flatnonzero(train['meta'].environment.isin(inner_held).to_numpy()&(train['meta'].horizon_s<=1).to_numpy())
        held=subset(train,held_ix)
        save_json(folder/'partition.json',dict(outer_training=outer['training_environments'],outer_held=outer['held_environments'],inner_fitting=inner_fit,inner_held=inner_held,inner_noise=inner_noise))
        screen=[]
        for p,b in GRID:
            tag=f'pg{p}_b{b}';checkpoint=folder/f'screen_{tag}.pt'
            fit_ix=choose_rows(train['meta'],inner_fit,p)
            model=load_completed(checkpoint)
            if model is None:
                model=fit(subset(train,fit_ix),inner_noise,SEEDS[0],b,args.updates,held,args.every,args.patience)
                save_model(checkpoint,model,dict(inner_fitting_environments=inner_fit,inner_held_environments=inner_held,per_group=p,fitting_rows=len(fit_ix),held_rows=len(held_ix)))
                np.savez_compressed(folder/f'screen_{tag}_rows.npz',fitting=fit_ix,held=held_ix)
            row=dict(fold=number,per_group=p,batch=b,selected_updates=model['selected_updates'],executed_updates=model['executed_updates'],score=model['selected_inner_error'],initial_score=model['initial_inner_error'],fitting_rows=len(fit_ix),held_rows=len(held_ix))
            screen.append(row);print(json.dumps(dict(stage='screen',**row)),flush=True)
        ranking=pd.DataFrame(screen);ranking['rank']=ranking.score.rank(method='min');ranking.to_csv(folder/'screen_comparison.csv',index=False)
        all_screen.append(ranking)
        if args.screen_only:
            save_json(args.out/'fold0_screen_complete.json',dict(rows=screen,elapsed_s=time.perf_counter()-started));return
        best=min(screen,key=lambda r:(r['score'],-(float('inf') if r['per_group']==0 else r['per_group']),r['batch']))
        p,b=int(best['per_group']),int(best['batch'])
        save_json(folder/'chosen_configuration.json',best)
        outer_fit_ix=choose_rows(train['meta'],outer['training_environments'],p)
        outer_held_ix=np.flatnonzero(train['meta'].environment.isin(outer['held_environments']).to_numpy())
        outer_fit,outer_held=subset(train,outer_fit_ix),subset(train,outer_held_ix)
        np.savez_compressed(folder/'outer_rows.npz',fitting=outer_fit_ix,held=outer_held_ix)
        predictions=[]
        for seed in SEEDS:
            selected_path=folder/f'inner_seed{seed}.pt';selection=load_completed(selected_path)
            if selection is None:
                selection=load_completed(folder/f'screen_pg{p}_b{b}.pt') if seed==SEEDS[0] else fit(subset(train,choose_rows(train['meta'],inner_fit,p)),inner_noise,seed,b,args.updates,held,args.every,args.patience)
                save_model(selected_path,selection,dict(per_group=p,inner_fitting_environments=inner_fit,inner_held_environments=inner_held))
            path=folder/f'seed{seed}.pt';model=load_completed(path)
            if model is None:
                model=fit(outer_fit,outer['noise'],seed,b,selection['selected_updates'])
                save_model(path,model,dict(per_group=p,fitting_environments=outer['training_environments'],held_environments=outer['held_environments'],fitting_rows=len(outer_fit_ix)))
            prediction=predict(outer_held,model);predictions.append(prediction)
            summaries.append(save_predictions(folder,outer_held,prediction,outer['noise'],seed,number,'train_oof'))
        summaries.append(save_predictions(folder,outer_held,np.mean(predictions,axis=0),outer['noise'],-1,number,'train_oof'))
        save_json(folder/'complete.json',dict(configuration=best,rows=len(outer_held_ix),elapsed_s=time.perf_counter()-started))
        print(json.dumps(dict(stage='outer_complete',fold=number,rows=len(outer_held_ix),elapsed_s=time.perf_counter()-started)),flush=True)
        del held,outer_fit,outer_held,predictions
    screen=pd.concat(all_screen,ignore_index=True);screen.to_csv(args.out/'all_screen_comparison.csv',index=False)
    ranked=screen.groupby(['per_group','batch'],as_index=False)['rank'].mean()
    chosen=min(ranked.to_dict('records'),key=lambda r:(r['rank'],-(float('inf') if r['per_group']==0 else r['per_group']),r['batch']))
    p,b=int(chosen['per_group']),int(chosen['batch']);folder=args.out/'full';folder.mkdir(exist_ok=True)
    save_json(folder/'chosen_configuration.json',dict(**chosen,rule=identity['full_configuration_rule'],official_validation_inspected=False))
    inner_fit=folds[0]['training_environments'];inner_held=folds[0]['held_environments'];noise=estimate_noise([r for r in records if r.environment in inner_fit]);full_noise=estimate_noise(records)
    fit_ix=choose_rows(train['meta'],inner_fit,p);held_ix=np.flatnonzero(train['meta'].environment.isin(inner_held).to_numpy()&(train['meta'].horizon_s<=1).to_numpy())
    full_ix=choose_rows(train['meta'],sorted(train['meta'].environment.unique()),p)
    np.savez_compressed(folder/'selection_rows.npz',inner_fitting=fit_ix,inner_held=held_ix,full_fitting=full_ix)
    checkpoints=[]
    for seed in SEEDS:
        selected_path=folder/f'inner_seed{seed}.pt';selected=load_completed(selected_path)
        if selected is None:
            selected=fit(subset(train,fit_ix),noise,seed,b,args.updates,subset(train,held_ix),args.every,args.patience)
            save_model(selected_path,selected,dict(per_group=p,inner_fitting_environments=inner_fit,inner_held_environments=inner_held))
        path=folder/f'seed{seed}.pt';model=load_completed(path)
        if model is None:
            model=fit(subset(train,full_ix),full_noise,seed,b,selected['selected_updates'])
            save_model(path,model,dict(per_group=p,fitting_environments=sorted(train['meta'].environment.unique()),fitting_rows=len(full_ix),official_validation_used_for_selection=False))
        checkpoints.append(model)
    save_json(folder/'models_frozen_before_validation.json',dict(checkpoints={str(folder/f'seed{s}.pt'):sha(folder/f'seed{s}.pt') for s in SEEDS},configuration=chosen))
    validation=load_pack(args.data/'validation');predictions=[]
    for seed,model in zip(SEEDS,checkpoints):
        pvalue=predict(validation,model);predictions.append(pvalue);summaries.append(save_predictions(folder,validation,pvalue,full_noise,seed,-1,'validation'))
    summaries.append(save_predictions(folder,validation,np.mean(predictions,axis=0),full_noise,-1,-1,'validation'))
    save_json(folder/'complete.json',dict(configuration=chosen,rows=len(validation['meta']),elapsed_s=time.perf_counter()-started))
    pd.concat(summaries,ignore_index=True).to_csv(args.out/'comparison.csv',index=False)
    save_json(args.out/'manifest.json',dict(**identity,elapsed_s=time.perf_counter()-started,final_configuration=chosen))


if __name__=='__main__':
    main()
