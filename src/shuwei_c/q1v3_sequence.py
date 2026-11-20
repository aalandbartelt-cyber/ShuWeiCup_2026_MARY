"""Shared-hand causal GRU around a coherent four-state probability prior."""
import numpy as np
import torch
from .q1v2_quality import weights


def history(frame,queries,raw,lags=(.7,.6,.5,.4,.3,.2,.1,0.)):
    """Only source rows at or before each requested historical time are read."""
    out=np.zeros((len(queries),2,len(lags),raw.shape[-1]),np.float32)
    known=np.zeros((len(queries),2,len(lags),1),np.float32)
    for episode,ix in queries.groupby('episode',sort=False).groups.items():
        source=frame[frame.episode==episode].sort_values('time_s');indices=source.index.to_numpy();t=source.time_s.to_numpy()
        if not len(t):continue
        times=queries.loc[ix,'time_s'].to_numpy()
        for k,lag in enumerate(lags):
            take=np.searchsorted(t,times-lag,side='right')-1;valid=take>=0
            chosen=np.asarray(ix)[valid];out[chosen,:,k]=raw[indices[take[valid]]];known[chosen,:,k]=1
    return out,known


class Net(torch.nn.Module):
    def __init__(self,features,hidden=16):
        super().__init__();self.gru=torch.nn.GRU(features,hidden,batch_first=True)
        self.output=torch.nn.Sequential(torch.nn.Dropout(.3),torch.nn.Linear(hidden,1))
    def forward(self,x):
        n,h,t,d=x.shape;_,hidden=self.gru(x.reshape(n*h,t,d))
        return self.output(hidden[0]).reshape(n,h)


def probabilities(net,x,prior):
    delta=2*torch.tanh(net(x));logits=torch.stack([torch.zeros(len(x),device=x.device),delta[:,0],delta[:,1],delta.sum(dim=1)],dim=1)
    return torch.softmax(prior+logits,dim=1),delta


def standardized(raw,known,mean,std):
    return np.concatenate([np.clip((raw-mean)/std,-4,4)*known,known],axis=-1).astype(np.float32)


def fit(frame,raw,known,prior,reg=.001,seed=20261120,epochs=20):
    torch.manual_seed(seed);w=weights(frame);wm=w[:,None,None,None]*known;total=wm.sum()
    mean=(raw*wm).sum(axis=(0,1,2))/total;std=np.maximum(np.sqrt((((raw-mean)**2)*wm).sum(axis=(0,1,2))/total),1e-4)
    x=torch.tensor(standardized(raw,known,mean,std),device='cuda');lp=torch.tensor(np.log(np.clip(prior,1e-6,1)),device='cuda',dtype=torch.float32)
    target=frame.target_state.to_numpy();y=torch.tensor(np.c_[target>0,target%2,target//2],device='cuda',dtype=torch.float32)
    w=torch.tensor(w*len(w),device='cuda',dtype=torch.float32);lw=torch.tensor([.5,.25,.25],device='cuda')
    net=Net(x.shape[-1]).cuda();torch.nn.init.zeros_(net.output[-1].weight);torch.nn.init.zeros_(net.output[-1].bias)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.05)
    for epoch in range(epochs):
        net.train()
        for ix in torch.randperm(len(w),device='cuda').split(2048):
            p,delta=probabilities(net,x[ix],lp[ix]);prob=torch.stack([1-p[:,0],p[:,1]+p[:,3],p[:,2]+p[:,3]],dim=1)
            loss=((((prob-y[ix])**2)*lw).sum(dim=1)*w[ix]).mean()+reg*delta.square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);optimizer.step()
    return dict(state_dict=net.cpu().state_dict(),mean=mean,std=std,regularization=reg,seed=seed,epochs=epochs)


def predict(raw,known,prior,model):
    net=Net(raw.shape[-1]+1).cuda();net.load_state_dict(model['state_dict']);net.eval();parts=[]
    with torch.inference_mode():
        for i in range(0,len(raw),2048):
            sl=slice(i,i+2048);x=torch.tensor(standardized(raw[sl],known[sl],model['mean'],model['std']),device='cuda')
            lp=torch.tensor(np.log(np.clip(prior[sl],1e-6,1)),device='cuda',dtype=torch.float32)
            parts.append(probabilities(net,x,lp)[0].cpu().numpy())
    return np.concatenate(parts)
