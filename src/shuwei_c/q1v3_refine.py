"""Exact-gradient/Hessian refinement for the convex displacement initializer.

AI-assisted by Codex, 2026-11-20. Keeps the original objective and box bounds.
"""
import numpy as np
from .q1v3_state import row_weights,DATASETS


def distance_objective(coef,x,y,w,code,penalty,hessian=False):
    groups,k=coef.shape;prior=np.zeros_like(coef);prior[:,0]=1.
    grad=np.zeros_like(coef);hess=np.zeros((groups*k,groups*k));value=0.
    for group in range(groups):
        ix=np.flatnonzero(code==group) if groups>1 else np.arange(len(x))
        for take in np.array_split(ix,max(1,int(np.ceil(len(ix)/4096)))):
            a=x[take];e=np.einsum('nck,k->nc',a,coef[group])-y[take];r=np.sqrt((e**2).sum(axis=1)+1e-8);weight=w[take]
            v=np.einsum('nck,nc->nk',a,e);value+=np.dot(weight,r);grad[group]+=(v*(weight/r)[:,None]).sum(axis=0)
            if hessian:
                block=np.einsum('nck,ncl,n->kl',a,a,weight/r)-np.einsum('nk,nl,n->kl',v,v,weight/r**3)
                s=slice(group*k,(group+1)*k);hess[s,s]+=block
    value+=penalty*np.mean((coef-prior)**2);grad+=2*penalty*(coef-prior)/coef.size
    if hessian:hess+=np.eye(coef.size)*2*penalty/coef.size
    if groups>1:
        delta=coef-coef.mean(axis=0);value+=penalty*np.mean(delta**2);grad+=2*penalty*delta/coef.size
        if hessian:hess+=np.kron(np.eye(groups)-np.ones((groups,groups))/groups,np.eye(k))*2*penalty/coef.size
    return float(value),grad.ravel(),hess


def refine(pack,initial,maxiter=80):
    x=pack['experts'].astype(float);y=pack['target'].astype(float);w=row_weights(pack['meta']);code=np.array([DATASETS.index(s) for s in pack['meta'].dataset]);coef=np.array(initial['coefficients'],float);shape=coef.shape;penalty=initial['penalty'];trace=[]
    for iteration in range(maxiter):
        value,grad,hess=distance_objective(coef,x,y,w,code,penalty,True);flat=coef.ravel();projected=grad.copy()
        active=((flat<=-2+1e-8)&(grad>0))|((flat>=2-1e-8)&(grad<0));projected[active]=0.;norm=float(abs(projected).max())
        trace.append(dict(iteration=iteration,objective=value,projected_gradient_inf=norm))
        if norm<1e-7:break
        free=~active;step=np.zeros_like(flat);step[free]=-np.linalg.solve(hess[np.ix_(free,free)]+np.eye(free.sum())*1e-10,grad[free])
        rate=1.
        for j in np.flatnonzero(step):rate=min(rate,((2-flat[j])/step[j] if step[j]>0 else (-2-flat[j])/step[j]))
        rate=max(0,rate)
        while rate>1e-10:
            candidate=np.clip(flat+rate*step,-2,2).reshape(shape);new_value,_,_=distance_objective(candidate,x,y,w,code,penalty)
            if new_value<=value+1e-4*rate*np.dot(grad,step):break
            rate*=.5
        if rate<=1e-10:break
        coef=candidate
    value,grad,_=distance_objective(coef,x,y,w,code,penalty);flat=coef.ravel();grad[((flat<=-2+1e-8)&(grad>0))|((flat>=2-1e-8)&(grad<0))]=0
    result=dict(initial,coefficients=coef.tolist(),objective=value,convergence=bool(abs(grad).max()<1e-7),iterations=len(trace),projected_gradient_inf=float(abs(grad).max()),optimizer='box-constrained Newton with exact Hessian and Armijo backtracking',trace=trace)
    return result
