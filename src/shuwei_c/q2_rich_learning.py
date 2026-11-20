"""Fold-local age-conditioned distance fitting and bounded residual gates."""
import time
import numpy as np
from scipy.optimize import minimize
import torch
from .q2_learning import normalized,row_weights
from .q2_features import DATASETS

KNOTS=np.array([0.,1/30,.1,.2,.5,1.,2.])

def interpolation(meta):
    age=np.log1p(np.clip(meta.age_s.to_numpy(float),0.,2.));knots=np.log1p(KNOTS)
    right=np.searchsorted(knots,age,side='right').clip(1,len(knots)-1);left=right-1
    fraction=(age-knots[left])/(knots[right]-knots[left])
    ds=np.array([DATASETS.index(d) for d in meta.dataset])
    return ds*len(KNOTS)+left,ds*len(KNOTS)+right,fraction

def prior_weights(meta,prior):
    left,right,fraction=interpolation(meta);c=np.asarray(prior['coefficients'],np.float32)
    return ((1-fraction[:,None])*c[left]+fraction[:,None]*c[right]).astype(np.float32)

def fit_prior(pack,noise,fitted=True,penalty=.001):
    raw=normalized(pack,noise);x=raw['e'].astype(float);y=raw['y'].astype(float);w=raw['weights']
    k=x.shape[-1];prior=np.zeros((3*len(KNOTS),k));prior[:,8]=1.
    left,right,fraction=interpolation(pack['meta']);n=len(prior)
    def objective(flat):
        c=flat.reshape(n,k);a=(1-fraction[:,None])*c[left]+fraction[:,None]*c[right]
        residual=np.einsum('nck,nk->nc',x,a)-y;distance=np.sqrt(np.sum(residual**2,axis=1)+1e-8)
        rows=np.einsum('nck,nc->nk',x,residual)*(w/distance)[:,None]
        grad=np.zeros_like(c)
        np.add.at(grad,left,rows*(1-fraction[:,None]));np.add.at(grad,right,rows*fraction[:,None])
        value=float(w@distance)+penalty*np.mean((c-prior)**2)
        grad+=2*penalty*(c-prior)/c.size
        # Smooth adjacent age knots, independently within each declared dataset.
        shape=c.reshape(3,len(KNOTS),k);delta=np.diff(shape,axis=1)
        value+=penalty*np.mean(delta**2);smooth=2*penalty*delta/delta.size
        g=grad.reshape(shape.shape);g[:,:-1]-=smooth;g[:,1:]+=smooth
        return value,grad.ravel()
    start=time.perf_counter()
    result=minimize(objective,prior.ravel(),jac=True,method='L-BFGS-B',bounds=[(-2.,2.)]*prior.size,
                    options=dict(maxiter=180,ftol=1e-10,gtol=1e-7)) if fitted else None
    return dict(coefficients=(result.x.reshape(prior.shape) if fitted else prior).tolist(),
                knots=KNOTS.tolist(),fitted=fitted,penalty=penalty,rows=len(x),
                objective=float(result.fun) if fitted else objective(prior.ravel())[0],
                convergence=bool(result.success) if fitted else True,
                iterations=int(result.nit) if fitted else 0,elapsed_s=time.perf_counter()-start,
                method='scipy.optimize.minimize L-BFGS-B; interpolated dataset/age coefficients')

def network(features,experts):
    net=torch.nn.Sequential(torch.nn.Linear(features,48),torch.nn.SiLU(),torch.nn.Linear(48,48),torch.nn.SiLU(),torch.nn.Linear(48,experts))
    torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
    return net

def tensors(pack,noise,mean,std,prior):
    raw=normalized(pack,noise)
    def tensor(a):return torch.as_tensor(a,device='cuda',dtype=torch.float32)
    return dict(x=tensor(np.clip((raw['x']-mean)/std,-10,10)),e=tensor(raw['e']),y=tensor(raw['y']),
                w=tensor(raw['weights']*len(raw['weights'])),base=tensor(prior_weights(pack['meta'],prior)))

def prediction(net,d,ix):
    delta=.5*torch.tanh(net(d['x'][ix]));weights=d['base'][ix]+delta
    return torch.einsum('nck,nk->nc',d['e'][ix],weights),delta

def held_score(net,d):
    net.eval();result=0.
    with torch.inference_mode():
        for start in range(0,len(d['x']),16384):
            sl=slice(start,start+16384);p,_=prediction(net,d,sl)
            result+=float((torch.linalg.vector_norm(p-d['y'][sl],dim=1)*d['w'][sl]).sum())
    return result/len(d['x'])

def fit_gate(train,noise,prior,held=None,seed=20261120,updates=800,batch=1024,every=25,patience=8):
    torch.manual_seed(seed);raw=normalized(train,noise);w=raw['weights']
    mean=np.average(raw['x'],weights=w,axis=0);std=np.maximum(np.sqrt(np.average((raw['x']-mean)**2,weights=w,axis=0)),1e-4)
    d=tensors(train,noise,mean,std,prior);v=tensors(held,noise,mean,std,prior) if held is not None else None
    net=network(len(mean),train['experts'].shape[-1]).cuda();optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    rng=torch.Generator(device='cuda');rng.manual_seed(seed)
    best=held_score(net,v) if held is not None else float('inf');initial=best
    state={k:x.detach().cpu().clone() for k,x in net.state_dict().items()};chosen=0;count=0;stale=0;seen=0
    trace=[dict(updates=0,held_error=best if held is not None else None,examples_seen=0)]
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    while count<updates:
        for ix in torch.randperm(len(d['x']),generator=rng,device='cuda').split(batch):
            net.train();p,delta=prediction(net,d,ix)
            loss=(torch.sqrt((p-d['y'][ix]).square().sum(1)+1e-8)*d['w'][ix]).mean()+.001*delta.square().mean()
            assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);optimizer.step()
            count+=1;seen+=len(ix)
            if count%every==0 or count==updates:
                score=held_score(net,v) if held is not None else None
                trace.append(dict(updates=count,held_error=score,examples_seen=seen))
                if held is None or score<best-1e-6:
                    if held is not None:best=score
                    state={k:x.detach().cpu().clone() for k,x in net.state_dict().items()};chosen=count;stale=0
                else:stale+=1
            if count>=updates or (held is not None and stale>=patience):break
        if held is not None and stale>=patience:break
    result=dict(state_dict=state,mean=mean,std=std,noise=noise,prior=prior,feature_names=train['feature_names'],expert_names=train['expert_names'],
                seed=seed,selected_updates=chosen,executed_updates=count,initial_held_error=initial if held is not None else None,
                selected_held_error=best if held is not None else None,trace=trace,elapsed_s=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),batch_size=batch,optimizer='torch.optim.AdamW',lr=.001,weight_decay=.01)
    del d,v,net,optimizer,raw;torch.cuda.empty_cache();return result

def predict(pack,model,base_only=False):
    raw=normalized(pack,model['noise']);base=prior_weights(pack['meta'],model['prior']);e=raw['e']
    if base_only:return np.einsum('nck,nk->nc',e,base)*raw['scales'][:,None]
    net=network(len(model['feature_names']),len(model['expert_names'])).cuda();net.load_state_dict(model['state_dict']);net.eval();parts=[]
    with torch.inference_mode():
        for start in range(0,len(e),16384):
            sl=slice(start,start+16384);x=torch.tensor(np.clip((raw['x'][sl]-model['mean'])/model['std'],-10,10),device='cuda',dtype=torch.float32)
            weights=base[sl]+.5*torch.tanh(net(x)).cpu().numpy();parts.append(np.einsum('nck,nk->nc',e[sl],weights)*raw['scales'][sl,None])
    del net;torch.cuda.empty_cache();return np.concatenate(parts)
