"""Shared-hand neural probability correction with coherent joint probabilities."""
import numpy as np
import torch
from .q1v2_quality import weights


def network(features):
    return torch.nn.Sequential(torch.nn.Linear(features,32),torch.nn.SiLU(),
        torch.nn.Dropout(.3),torch.nn.Linear(32,1))


def probability(net,x,log_prior):
    delta=2*torch.tanh(net(x).squeeze(-1))
    logits=torch.stack([torch.zeros(len(x),device=x.device),delta[:,0],delta[:,1],delta.sum(dim=1)],dim=1)
    return torch.softmax(log_prior+logits,dim=1),delta


def fit(frame,raw,prior,reg=.001,seed=20261120,epochs=20):
    torch.manual_seed(seed);w=weights(frame)
    mean=np.average(raw.mean(axis=1),weights=w,axis=0)
    std=np.maximum(np.sqrt(np.average(((raw-mean)**2).mean(axis=1),weights=w,axis=0)),1e-4)
    x=torch.tensor(np.clip((raw-mean)/std,-4,4),device='cuda',dtype=torch.float32)
    lp=torch.tensor(np.log(np.clip(prior,1e-6,1)),device='cuda',dtype=torch.float32)
    target=frame.target_state.to_numpy();y=torch.tensor(np.c_[target>0,target%2,target//2],device='cuda',dtype=torch.float32)
    w=torch.tensor(w*len(w),device='cuda',dtype=torch.float32);lw=torch.tensor([.5,.25,.25],device='cuda')
    net=network(x.shape[-1]).cuda();torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.05)
    for epoch in range(epochs):
        net.train()
        for ix in torch.randperm(len(w),device='cuda').split(4096):
            p,delta=probability(net,x[ix],lp[ix]);prob=torch.stack([1-p[:,0],p[:,1]+p[:,3],p[:,2]+p[:,3]],dim=1)
            loss=((((prob-y[ix])**2)*lw).sum(dim=1)*w[ix]).mean()+reg*delta.square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);optimizer.step()
    return dict(state_dict=net.cpu().state_dict(),mean=mean,std=std,regularization=reg,seed=seed,epoch=epochs)


def predict(raw,prior,model):
    x=torch.tensor(np.clip((raw-model['mean'])/model['std'],-4,4),device='cuda',dtype=torch.float32)
    lp=torch.tensor(np.log(np.clip(prior,1e-6,1)),device='cuda',dtype=torch.float32)
    net=network(x.shape[-1]).cuda();net.load_state_dict(model['state_dict']);net.eval()
    with torch.inference_mode():return probability(net,x,lp)[0].cpu().numpy()
