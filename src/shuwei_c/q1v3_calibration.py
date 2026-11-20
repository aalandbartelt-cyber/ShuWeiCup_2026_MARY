"""Causal mature-residual summaries and empirical conditional prediction radii."""
import numpy as np
import pandas as pd
from .q1v3_state import row_weights
from .q1v2_state import weighted_quantile


def mature_residual_features(frame,error_column='gate_normalized_error'):
    out=frame.copy();constants=[.3,1.,3.]
    for tau in constants:out[f'past_residual_ewm_{tau}']=np.nan
    out['past_residual_known']=0.;out['past_residual_age']=0.
    for _,g in frame.groupby(['episode','hand'],sort=False):
        g=g.sort_values('origin_s');order=np.argsort(g.target_s.to_numpy());target=g.target_s.to_numpy()[order]
        residual=g[error_column].to_numpy()[order];pointer=0;last=None;averages=np.full(3,np.nan)
        for ix,row in g.iterrows():
            now=row.origin_s
            while pointer<len(target) and target[pointer]<=now+1e-9:
                if last is None:averages[:]=residual[pointer]
                else:
                    alpha=-np.expm1(-(target[pointer]-last)/np.array(constants));averages=(1-alpha)*averages+alpha*residual[pointer]
                last=target[pointer];pointer+=1
            if last is not None:
                for j,tau in enumerate(constants):out.loc[ix,f'past_residual_ewm_{tau}']=np.log1p(averages[j])
                out.loc[ix,'past_residual_known']=1.;out.loc[ix,'past_residual_age']=now-last
    return out


def conditional_columns(frame,online=False):
    names=[c for c in frame if c.startswith('x_')]
    if online:names += [c for c in frame if c.startswith('past_residual_')]
    return names


def calibration_shift(frame,prediction,level):
    residual=frame.gate_normalized_error.to_numpy()-prediction;w=row_weights(frame)
    global_shift=weighted_quantile(residual,w,level);specific={}
    for ds,g in frame.groupby('dataset'):
        take=(frame.dataset==ds).to_numpy();local=weighted_quantile(residual[take],row_weights(g),level)
        groups=g.environment.nunique();shrink=groups/(groups+3.)
        specific[ds]=float(shrink*local+(1-shrink)*global_shift)
    return dict(global_shift=global_shift,dataset_shift=specific)


def radius_metrics(frame,radius,level,method):
    rows=[]
    for ds in ['ALL',*frame.dataset.unique()]:
        take=np.ones(len(frame),bool) if ds=='ALL' else (frame.dataset==ds).to_numpy();f=frame.loc[take]
        error=f.gate_normalized_error.to_numpy();r=radius[take];w=row_weights(f)
        pinball=np.where(error>=r,level*(error-r),(1-level)*(r-error))
        for scale,label in [(np.ones(len(f)),'normalized'),(f.scale.to_numpy(),'native')]:
            rows.append(dict(dataset=ds,units=label,method=method,nominal=level,coverage=float(np.dot(w,error<=r)),
                radius=float(np.dot(w,r*scale)),pinball=float(np.dot(w,pinball*scale)),environments=f.environment.nunique(),n=len(f)))
    return rows
