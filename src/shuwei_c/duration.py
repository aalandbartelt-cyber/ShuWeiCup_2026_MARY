"""Weighted, regularized parametric duration and prefix-transition models.

AI: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
Only coefficients and covariance matrices reach policies, never a source library.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm
from .data import balanced_weights, prefix_features

FAMILIES = ('P','A','L')
LEVELS = (.2,.6,1.)


def design(features, families):
    return np.column_stack([np.log1p(np.maximum(features,0)),
                            np.array([[float(f==k) for k in FAMILIES] for f in families])])


def fit_ridge(x, y, w, penalty=10.):
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    scale = np.maximum(scale, .05)
    a = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    w = w/w.sum()*len(w)
    reg = np.eye(a.shape[1])*penalty
    reg[0,0] = 1e-8
    coef = np.linalg.solve(a.T@(w[:,None]*a)+reg, a.T@(w[:,None]*y))
    return dict(mean=mean.tolist(), scale=scale.tolist(), coef=coef.tolist())


def ridge_predict(model, x):
    x = np.atleast_2d(x)
    a = np.column_stack([np.ones(len(x)), (x-np.array(model['mean']))/np.array(model['scale'])])
    return a@np.array(model['coef'])


def fit_duration(data, use_quality=True, penalty=10.):
    units = data.eligible('train')
    env = np.array([u['environment_id'] for u in units])
    ep = np.array([u['episode_uuid'] for u in units])
    weights = balanced_weights(env,ep)
    family = [u['family_id'] for u in units]
    y = np.log([float(u['duration_s']) for u in units])[:,None]
    features = [np.array([prefix_features(data.unit_prefix(u,h),use_quality) for u in units]) for h in LEVELS]
    result = dict(version=1, use_quality=use_quality, penalty=penalty,
                  training_units=len(units), training_environments=len(set(env)), levels=[], transition=[])
    # Shared family shrinkage baseline. Family L has very few independent environments.
    prior_x = design(np.zeros((len(units),0)),family)
    for level in [-1,0,1,2]:
        x = prior_x if level==-1 else design(features[level],family)
        model = fit_ridge(x,y,weights,penalty)
        residuals = np.empty_like(y)
        for e in np.unique(env):
            held = env==e
            m = fit_ridge(x[~held],y[~held],balanced_weights(env[~held],ep[~held]),penalty)
            residuals[held] = y[held]-ridge_predict(m,x[held])
        model['sigma'] = float(np.sqrt(np.average(residuals[:,0]**2,weights=weights)))
        model['oof_log_rmse'] = model['sigma']
        if level==-1:
            result['prior'] = model
        else:
            result['levels'].append(model)
        if 0<=level<2:
            # Parametric transition, no nearest-neighbour or source matching.
            next_features = np.log1p(np.maximum(features[level+1],0))
            transition = fit_ridge(x,next_features,weights,penalty)
            errors = np.empty_like(next_features)
            for e in np.unique(env):
                held=env==e
                m=fit_ridge(x[~held],next_features[~held],balanced_weights(env[~held],ep[~held]),penalty)
                errors[held]=next_features[held]-ridge_predict(m,x[held])
            cov = (errors*weights[:,None]).T@errors
            # Shrink to diagonal to stabilize the small-environment covariance.
            cov = .5*cov+.5*np.diag(np.diag(cov))+np.eye(cov.shape[0])*1e-8
            transition['covariance'] = cov.tolist()
            result['transition'].append(transition)
    return result


class DurationModel:
    def __init__(self, params, use_prefix=True):
        self.params, self.use_prefix = params, use_prefix

    def distribution(self, family, prefix, level):
        if self.use_prefix:
            features = prefix_features(prefix,self.params['use_quality'])
            x=design(features[None,:],[family])
            m=self.params['levels'][level]
        else:
            x=design(np.zeros((1,0)),[family]); m=self.params['prior']
        mu=float(ridge_predict(m,x)[0,0])
        return mu,max(m['sigma'],.05)

    def hypothetical(self, family, prefix, level, count=8):
        if level>=2 or not self.use_prefix:
            return []
        f=prefix_features(prefix,self.params['use_quality'])
        x=design(f[None,:],[family])
        m=self.params['transition'][level]
        mean=ridge_predict(m,x)[0]
        # Common deterministic Monte Carlo draws, independent of replay/source seed.
        rng=np.random.default_rng(83021+level)
        draws=rng.multivariate_normal(mean,np.array(m['covariance']),size=count)
        next_model=self.params['levels'][level+1]
        out=[]
        for draw in draws:
            fnext=np.expm1(np.clip(draw,0,6))
            mu=float(ridge_predict(next_model,design(fnext[None,:],[family]))[0,0])
            out.append((mu,max(next_model['sigma'],.05)))
        return out


def service_samples(distribution, alpha, beta, n=19):
    mu,sigma=distribution
    # Eligible source duration is at least 1.2 s; truncated lognormal, not clipping mass.
    low=float(ndtr((np.log(1.2)-mu)/sigma))
    q=low+(1-low)*(np.arange(n)+.5)/n
    d=np.exp(np.clip(mu+sigma*norm.ppf(np.clip(q,1e-8,1-1e-8)),-10,10))
    return alpha*d+beta


def save_model(params,path):
    Path(path).write_text(json.dumps(params,indent=2,ensure_ascii=False)+'\n')
