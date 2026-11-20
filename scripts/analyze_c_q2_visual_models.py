#!/usr/bin/env python3
"""Paired environment audit of visual additions, averaging losses over seeds."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd


KEYS=['episode','environment','dataset','window_id','horizon_s','origin_s','time_s','hand','cap','delay_s','phase','scenario','age_s','current_direct']
CONDITION=['dataset','horizon_s','cap','delay_s','subset']
BRANCHES=('numeric','image','delayed','permuted')


def intervals(table,keys):
    rng=np.random.default_rng(20261120);rows=[]
    for key,group in table.groupby(keys,dropna=False,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        numeric=group.numeric.to_numpy();indices=rng.integers(0,len(group),(4000,len(group)))
        for branch in BRANCHES[1:]:
            value=group[branch].to_numpy();gain=1-value.mean()/numeric.mean()
            boot=1-value[indices].mean(axis=1)/numeric[indices].mean(axis=1)
            rows.append(dict(zip(keys,key))|dict(branch=branch,numeric=float(numeric.mean()),
                candidate=float(value.mean()),relative_improvement=float(gain),
                lower=float(np.quantile(boot,.025)),upper=float(np.quantile(boot,.975)),
                environments=len(group),improved_environments=int((value<numeric).sum())))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20261120,20261121,20261122])
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    env=[];files=[];seed_tables=[];seen=set()
    for fold in range(4):
        folder=args.root/f'fold{fold}';files.append(folder/'manifest.json')
        if not files[-1].exists():raise ValueError(f'Incomplete fold {fold}')
        reference=None;losses={branch:[] for branch in BRANCHES}
        for branch in BRANCHES:
            for seed in args.seeds:
                subfolder='numeric' if branch=='numeric' else 'image'
                suffix='' if branch in ('numeric','image') else f'_{branch}'
                file=folder/subfolder/f'seed{seed}{suffix}_errors.parquet';files.append(file)
                frame=pd.read_parquet(file)
                if reference is None:reference=frame[KEYS].copy()
                elif not reference.equals(frame[KEYS]):raise ValueError('Unpaired rows')
                losses[branch].append(frame.normalized_error.to_numpy())
                s=frame.groupby(['environment','episode','window_id','horizon_s','cap','delay_s']).normalized_error.mean().reset_index()
                s=s.groupby(['environment','episode','horizon_s','cap','delay_s']).normalized_error.mean().reset_index()
                s=s.groupby(['environment','horizon_s','cap','delay_s']).normalized_error.mean().reset_index()
                s['branch']=branch;s['seed']=seed;s['fold']=fold;seed_tables.append(s)
        environments=set(reference.environment.unique())
        if seen&environments:raise ValueError('Outer folds overlap')
        seen|=environments
        for branch in BRANCHES:reference[branch]=np.mean(losses[branch],axis=0)
        for subset,mask in [('all',np.ones(len(reference),bool)),('not_direct',~reference.current_direct.to_numpy())]:
            for dataset in ['ALL']+sorted(reference.dataset.unique()):
                take=mask if dataset=='ALL' else mask&reference.dataset.eq(dataset).to_numpy()
                f=reference.loc[take]
                window=f.groupby(['environment','episode','window_id','horizon_s','cap','delay_s'])[list(BRANCHES)].mean().reset_index()
                record=window.groupby(['environment','episode','horizon_s','cap','delay_s'])[list(BRANCHES)].mean().reset_index()
                table=record.groupby(['environment','horizon_s','cap','delay_s'])[list(BRANCHES)].mean().reset_index()
                table['dataset']=dataset;table['subset']=subset;table['fold']=fold;env.append(table)
    env=pd.concat(env,ignore_index=True);env.to_csv(args.out/'environment.csv',index=False)
    intervals(env,CONDITION).to_csv(args.out/'paired_conditions.csv',index=False)
    main=env[(env.horizon_s<=1)&env.delay_s.eq(.1)]
    macro=main.groupby(['dataset','subset','environment'])[list(BRANCHES)].mean().reset_index()
    intervals(macro,['dataset','subset']).to_csv(args.out/'paired_main.csv',index=False)
    pd.concat(seed_tables,ignore_index=True).to_csv(args.out/'seed_environment.csv',index=False)
    files.append(Path(__file__))
    (args.out/'manifest.json').write_text(json.dumps(dict(input_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        fold_count=4,seeds=args.seeds,environments=len(seen),bootstrap_draws=4000,
        interpretation='Mean loss over matched single-seed models; not prediction ensemble. Paired environment bootstrap is descriptive; folds share training data.',
        main='Equal main H and cap conditions at delay .1, then equal environments.',test_used=False),indent=2)+'\n')


if __name__=='__main__':main()
