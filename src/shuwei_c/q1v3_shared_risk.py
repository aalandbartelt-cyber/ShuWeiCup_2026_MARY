"""Parsimonious shared-hand log-odds correction around a joint transition prior.

AI-assisted by Codex, 2026-11-20. Head-view diagnostics remain interpretable;
the same coefficients apply to both hands. Probabilities remain coherent.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from .q1v2_quality import weights


def features(frame,diagnostics,video=True):
    names=['duration','speed','turn','radius','history_.1','history_.3','history_1','edge','depth'];numeric=[]
    for side in ['left','right']:
        columns=[f'duration_{side}',*[f'motion_{side}_{x}' for x in ['speed','turn','radius']],*[f'history_{side}_{w}' for w in [.1,.3,1.]]]
        x=frame[columns].to_numpy()
        for field in ['edge','depth']:x=np.c_[x,np.mean([frame[f'geometry_{v}_{side}_{field}'].to_numpy() for v in ['head_left','head_right']],axis=0)]
        numeric.append(x)
    x=np.stack(numeric,axis=1)
    if video:
        d=diagnostics.reshape(len(frame),2,4,3,32)[:,:,:2]
        ix=[12,15,16,17,18,21,22,24,26,29]
        current=d[:,:,:,0,:][...,ix].mean(axis=2);past=d[:,:,:,1,:][...,ix].mean(axis=2)
        x=np.concatenate([x,current,current-past],axis=-1)
        fields=['crop_available','log_sharpness','clipped_pixels','crop_area','boundary_distance','motion_median','motion_p90','detected','projection_disagreement','frame_age']
        names+=['video_'+s for s in fields]+['video_change_'+s for s in fields]
    flag=frame[['flag_left','flag_right']].to_numpy()
    return x,flag,names


def design(x,flag,mean,std):
    z=np.clip((x-mean)/std,-4,4)
    return np.concatenate([np.ones((*x.shape[:2],1)),flag[...,None],z,z*flag[...,None]],axis=-1)


def predict(x,flag,prior,model):
    z=design(x,flag,model['mean'],model['std']);delta=np.einsum('nhk,k->nh',z,model['coef'])
    logits=np.c_[np.zeros(len(x)),delta[:,0],delta[:,1],delta.sum(axis=1)]
    return softmax(np.log(np.clip(prior,1e-7,1))+logits,axis=1)


def fit(frame,x,flag,prior,penalty=.001,objective='brier'):
    w=weights(frame);mean=np.average(x.mean(axis=1),weights=w,axis=0)
    std=np.maximum(np.sqrt(np.average(((x-mean)**2).mean(axis=1),weights=w,axis=0)),1e-4);z=design(x,flag,mean,std)
    lp=np.log(np.clip(prior,1e-7,1));target=frame.target_state.to_numpy();onehot=np.eye(4)[target]
    a=(target>0).astype(float);left=target%2;right=target//2
    def fun(beta):
        delta=np.einsum('nhk,k->nh',z,beta);logits=lp+np.c_[np.zeros(len(x)),delta[:,0],delta[:,1],delta.sum(axis=1)];p=softmax(logits,axis=1)
        if objective=='logloss':
            value=-np.dot(w,np.log(np.maximum(p[np.arange(len(p)),target],1e-12)));gradient=p-onehot
        else:
            ea=1-p[:,0]-a;el=p[:,1]+p[:,3]-left;er=p[:,2]+p[:,3]-right
            value=np.dot(w,.5*ea**2+.25*el**2+.25*er**2)
            gp=np.c_[-ea,.5*el,.5*er,.5*(el+er)];gradient=p*(gp-(gp*p).sum(axis=1,keepdims=True))
        gd=np.c_[gradient[:,1]+gradient[:,3],gradient[:,2]+gradient[:,3]]
        grad=np.einsum('nhk,nh,n->k',z,gd,w)+2*penalty*beta
        return value+penalty*np.sum(beta**2),grad
    result=minimize(fun,np.zeros(z.shape[-1]),jac=True,method='L-BFGS-B',bounds=[(-2,2)]*z.shape[-1],options=dict(maxiter=120,ftol=1e-11,gtol=1e-8))
    return dict(coef=result.x,mean=mean,std=std,penalty=penalty,objective=objective,converged=bool(result.success),iterations=result.nit)
