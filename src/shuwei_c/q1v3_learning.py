"""Small invariant-feature neural gate, reusable across environment folds."""
import numpy as np
import torch
from .q1v3_state import DATASETS,row_weights


def network(features,experts):
    return torch.nn.Sequential(torch.nn.Linear(features,48),torch.nn.SiLU(),torch.nn.Linear(48,48),torch.nn.SiLU(),torch.nn.Linear(48,experts))


def base_coefficients(meta,base):
    c=np.asarray(base['coefficients'],np.float32)
    return c[[DATASETS.index(d) for d in meta.dataset]] if base['dataset_specific'] else np.repeat(c,len(meta),axis=0)


def fit_gate(pack,base,epochs=10,regularization=.001,seed=20261120):
    torch.manual_seed(seed);torch.set_num_threads(2)
    w=row_weights(pack['meta']);mean=np.average(pack['features'],weights=w,axis=0)
    std=np.maximum(np.sqrt(np.average((pack['features']-mean)**2,weights=w,axis=0)),1e-4)
    x=torch.tensor(np.clip((pack['features']-mean)/std,-10,10),device='cuda',dtype=torch.float32)
    e=torch.tensor(pack['experts'],device='cuda');y=torch.tensor(pack['target'],device='cuda')
    weights=torch.tensor(w*len(w),device='cuda',dtype=torch.float32);b=torch.tensor(base_coefficients(pack['meta'],base),device='cuda')
    net=network(x.shape[1],e.shape[-1]).cuda();torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    for epoch in range(epochs):
        for ix in torch.randperm(len(x),device='cuda').split(4096):
            delta=.5*torch.tanh(net(x[ix]));prediction=torch.einsum('nck,nk->nc',e[ix],b[ix]+delta)
            loss=(torch.sqrt(((prediction-y[ix])**2).sum(dim=1)+1e-8)*weights[ix]).mean()+regularization*(delta**2).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);optimizer.step()
    return dict(state_dict=net.cpu().state_dict(),mean=mean,std=std,base=base,feature_names=pack['feature_names'],seed=seed,epoch=epochs,regularization=regularization)


def predict_gate(pack,model):
    x=torch.tensor(np.clip((pack['features']-model['mean'])/model['std'],-10,10),device='cuda',dtype=torch.float32)
    e=torch.tensor(pack['experts'],device='cuda');b=torch.tensor(base_coefficients(pack['meta'],model['base']),device='cuda')
    net=network(x.shape[1],e.shape[-1]).cuda();net.load_state_dict(model['state_dict']);net.eval();parts=[]
    with torch.inference_mode():
        for start in range(0,len(x),4096):
            sl=slice(start,start+4096);parts.append(torch.einsum('nck,nk->nc',e[sl],b[sl]+.5*torch.tanh(net(x[sl]))).cpu().numpy())
    return np.concatenate(parts)
