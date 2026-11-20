#!/usr/bin/env python3
"""Compare stored numerical candidates on identical target rows and scales."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q2_learning import load_pack,subset,row_weights

KEYS=['episode','window_id','scenario','time_s','hand']
XYZ=['prediction_'+c for c in 'xyz']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('artifacts/c_q2/point_comparison'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    pack=load_pack('artifacts/c_q2/learning_data/train')
    fp=Path('artifacts/c_q2/protocol/folds.json');folds=json.loads(fp.read_text());files=[fp,Path(__file__)]
    tables=[];environment=[];age_tables=[];main_rows=[]
    for fold in folds:
        n=fold['fold'];p=subset(pack,pack['meta'].environment.isin(fold['held_environments']).to_numpy());meta=p['meta']
        paths=dict(static=Path(f'artifacts/c_q2/static_gate/nested/fold{n}/ensemble_predictions.parquet'),
                   residual=Path(f'artifacts/c_q2/gate_oof_main/residual/fold{n}/predictions.parquet'),
                   rich_all=Path(f'artifacts/c_q2/rich_state/nested/fold{n}/ensemble_predictions.parquet'),
                   rich_main=Path(f'artifacts/c_q2/rich_state/main_horizons/fold{n}/ensemble_predictions.parquet'))
        pred={}
        for name,path in paths.items():
            frame=pd.read_parquet(path);files.append(path)
            if not frame[KEYS].equals(meta[KEYS]):raise ValueError(f'Row mismatch {name} {n}')
            pred[name]=frame[XYZ].to_numpy(float)
        pred['fixed_blend']=(pred['static']+pred['residual'])/2
        target=p['anchor'].astype(float)+p['target'].astype(float)
        scales=meta.dataset.map({d:x['hand_scale'] for d,x in fold['noise'].items()}).to_numpy()
        result=meta.copy()
        for name,value in pred.items():result[name]=np.linalg.norm(value-target,axis=1)/scales
        methods=list(pred)
        for scope,mask in [('main_all_delays',meta.horizon_s.le(1)),('main_primary_delay',meta.horizon_s.le(1)&meta.delay_s.eq(.1)),('stress_all_delays',meta.horizon_s.gt(1))]:
            group=result.loc[mask]
            for ds in ['ALL']+sorted(group.dataset.unique()):
                g=group if ds=='ALL' else group[group.dataset==ds];w=row_weights(g)
                for method in methods:main_rows.append(dict(fold=n,scope=scope,dataset=ds,method=method,error=float(w@g[method].to_numpy())))
        for ds in ['ALL']+sorted(result.dataset.unique()):
            f=result if ds=='ALL' else result[result.dataset==ds]
            w=f.groupby(['environment','episode','window_id','horizon_s','cap','delay_s'])[methods].mean().reset_index()
            r=w.groupby(['environment','episode','horizon_s','cap','delay_s'])[methods].mean().reset_index()
            e=r.groupby(['environment','horizon_s','cap','delay_s'])[methods].mean().reset_index();e['dataset']=ds;e['fold']=n;environment.append(e)
        group=result.copy();group['age_bin']=pd.cut(group.age_s,[-1e-9,.05,.1,.2,.5,1,2.1])
        for (ds,age),g in group.groupby(['dataset','age_bin'],observed=True):
            w=row_weights(g)
            for method in methods:age_tables.append(dict(fold=n,dataset=ds,age_bin=str(age),method=method,error=float(w@g[method].to_numpy()),rows=len(g)))
    main=pd.DataFrame(main_rows);main.to_csv(args.out/'fold_scopes.csv',index=False)
    main.groupby(['scope','dataset','method']).error.mean().reset_index().to_csv(args.out/'scope_summary.csv',index=False)
    env=pd.concat(environment,ignore_index=True);env.to_csv(args.out/'environment_conditions.csv',index=False)
    env.groupby(['dataset','horizon_s','cap','delay_s'])[methods].mean().reset_index().to_csv(args.out/'conditions.csv',index=False)
    pd.DataFrame(age_tables).to_csv(args.out/'age_diagnostics.csv',index=False)
    # Bootstrap whole environments after equal-weight main H/cap aggregation.
    primary=env[env.horizon_s.le(1)&env.delay_s.eq(.1)].groupby(['dataset','environment'])[methods].mean().reset_index()
    rng=np.random.default_rng(20261120);rows=[]
    for ds,g in primary.groupby('dataset'):
        idx=rng.integers(0,len(g),(4000,len(g)));base=g.fixed_blend.to_numpy()
        for method in methods:
            x=g[method].to_numpy();boot=1-x[idx].mean(axis=1)/base[idx].mean(axis=1)
            rows.append(dict(dataset=ds,method=method,error=float(x.mean()),relative_improvement=float(1-x.mean()/base.mean()),
                lower=float(np.quantile(boot,.025)),upper=float(np.quantile(boot,.975)),environments=len(g),improved_environments=int((x<base).sum())))
    pd.DataFrame(rows).to_csv(args.out/'primary_paired.csv',index=False)
    (args.out/'manifest.json').write_text(json.dumps(dict(input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        point_targets='original numerical learning cache anchor+target',fitting='stored whole-environment outer predictions',
        primary='delay .1, H<=1, equal conditions and environments',test_used=False,validation_used=False),indent=2)+'\n')


if __name__=='__main__':main()
