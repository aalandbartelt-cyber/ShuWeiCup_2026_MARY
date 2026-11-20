"""Small displacement correction trained against OOF numerical predictions."""
import numpy as np
import torch
from .q1v3_learning import network
from .q1v3_state import row_weights


def fit(pack,raw,prior,seed=20261120,epochs=5,regularization=.1):
    torch.manual_seed(seed);w=row_weights(pack['meta']);mean=np.average(raw,weights=w,axis=0)
    std=np.maximum(np.sqrt(np.average((raw-mean)**2,weights=w,axis=0)),1e-4)
    x=torch.tensor(np.clip((raw-mean)/std,-8,8),device='cuda',dtype=torch.float32)
    e=torch.tensor(pack['experts'],device='cuda');y=torch.tensor(pack['target'],device='cuda')
    b=torch.tensor(prior,device='cuda',dtype=torch.float32);w=torch.tensor(w*len(w),device='cuda',dtype=torch.float32)
    net=network(x.shape[1],e.shape[-1]).cuda();torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.05)
    for epoch in range(epochs):
        for ix in torch.randperm(len(w),device='cuda').split(4096):
            delta=.25*torch.tanh(net(x[ix]));p=b[ix]+torch.einsum('nck,nk->nc',e[ix],delta)
            loss=(torch.sqrt(((p-y[ix])**2).sum(dim=1)+1e-8)*w[ix]).mean()+regularization*delta.square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);optimizer.step()
    return dict(state_dict=net.cpu().state_dict(),mean=mean,std=std,seed=seed,epochs=epochs,regularization=regularization)


def predict(pack,raw,prior,model):
    x=torch.tensor(np.clip((raw-model['mean'])/model['std'],-8,8),device='cuda',dtype=torch.float32)
    e=torch.tensor(pack['experts'],device='cuda');net=network(x.shape[1],e.shape[-1]).cuda();net.load_state_dict(model['state_dict']);net.eval()
    with torch.inference_mode():delta=np.concatenate([torch.einsum('nck,nk->nc',e[i:i+4096],.25*torch.tanh(net(x[i:i+4096]))).cpu().numpy() for i in range(0,len(x),4096)])
    return prior+delta
