#!/usr/bin/env python3
"""Conservative radius floor for already flagged current hand observations.

The rare current-flag subgroup is fitted only from training OOF residuals;
outer folds use their own nested inner residuals. No future flag is an input.
"""
from pathlib import Path
import json,sys,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q1v3_state import row_weights
from shuwei_c.q1v2_state import weighted_quantile
from shuwei_c.q1v3_calibration import radius_metrics
from shuwei_c.q1v3_feedback import adapt_radii


def floor_parameters(frame,levels):
    f=frame[frame.x_flag_now>0]
    return dict(levels=levels,quantiles=[weighted_quantile(f.gate_normalized_error,row_weights(f),q) for q in levels],n=len(f),environments=f.environment.nunique())


def apply_floor(frame,base,parameters):
    out=base.copy();flag=frame.x_flag_now.to_numpy()>0
    out[flag]=np.maximum(out[flag],parameters['quantiles']);return out


def main():
    root=Path('artifacts/c_q1_v3');out=root/'flagged_calibration';out.mkdir(exist_ok=True);levels=[.5,.8,.9,.95]
    outer=pd.read_parquet(root/'calibration_nested/train_outer.parquet');base=np.load(root/'calibration_nested/radii.npz')['0.0'];corrected=np.zeros_like(base);spec={}
    for fold,g in outer.groupby('fold'):
        inner=pd.read_parquet(root/f'calibration_nested/fold{fold}_inner_residuals.parquet');parameters=floor_parameters(inner,levels)
        assert not set(inner.environment)&set(g.environment)
        corrected[g.index]=apply_floor(g,base[g.index],parameters);spec[str(fold)]=parameters
    train=pd.read_parquet(root/'calibration_screen/train_features.parquet');val=pd.read_parquet(root/'calibration_screen/validation_features.parquet')
    full=floor_parameters(train,levels);vb=np.load(root/'feedback_screen/validation_radii.npz')['0.0'];vc=apply_floor(val,vb,full);rows=[];groups=[];radii={}
    for split,frame,base,floored in [('train_outer',outer,base,corrected),('validation',val,vb,vc)]:
        for floor,array in [(False,base),(True,floored)]:
            for rate in [0.,.005,.01,.02]:
                r,_=adapt_radii(frame,array,levels,rate);name=f'flag_floor_{floor}_rate_{rate}';radii[split+'_'+name]=r
                for condition in ['all','current_flagged','current_good']:
                    mask=np.ones(len(frame),bool) if condition=='all' else ((frame.x_flag_now.to_numpy()>0) if condition=='current_flagged' else frame.x_flag_now.to_numpy()==0)
                    for k,level in enumerate(levels):
                        metrics=radius_metrics(frame.loc[mask].reset_index(drop=True),r[mask,k],level,name)
                        for item in metrics:item.update(split=split,condition=condition,flag_floor=floor,rate=rate)
                        rows+=metrics
                for env,g in frame.groupby('environment'):
                    metrics=radius_metrics(g.reset_index(drop=True),r[g.index,2],.9,name)
                    for item in metrics:item.update(split=split,environment=env,flag_floor=floor,rate=rate)
                    groups+=metrics
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(groups).to_csv(out/'environment_metrics.csv',index=False);np.savez_compressed(out/'radii.npz',**radii)
    (out/'parameters.json').write_text(json.dumps(dict(outer_folds=spec,full_training=full),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(test_used=False,scope='Radius floor equals same-level empirical quantile among currently flagged training OOF residuals; outer-held environments excluded from all floor estimates. Subgroup is small; no conditional coverage guarantee.',nesting='Input base is already cumulatively nested, before optional floor and delayed feedback; issued radii are nested again. Compare floor True and False within this experiment, not to the earlier raw-base feedback screen.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
