"""Joint-quality residual network with an explicit transition prior."""
import numpy as np
import torch
from .q1v2_quality import weights


def network(dim):
    return torch.nn.Sequential(torch.nn.Linear(dim,48),torch.nn.SiLU(),torch.nn.Dropout(.2),torch.nn.Linear(48,24),torch.nn.SiLU(),torch.nn.Linear(24,4))


def fit(frame,feature,prior,seed,epochs=10,penalty=.001):
    torch.manual_seed(seed);w=weights(frame);mean=np.average(feature,weights=w,axis=0)
    std=np.maximum(np.sqrt(np.average((feature-mean)**2,weights=w,axis=0)),1e-4)
    x=torch.tensor(np.clip((feature-mean)/std,-8,8),device='cuda',dtype=torch.float32)
    lp=torch.tensor(np.log(np.clip(prior,1e-6,1)),device='cuda',dtype=torch.float32)
    target=frame.target_state.to_numpy();y=torch.tensor(np.c_[target>0,target%2,target//2],device='cuda',dtype=torch.float32)
    w=torch.tensor(w*len(w),device='cuda',dtype=torch.float32);net=network(x.shape[1]).cuda()
    torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.05)
    loss_weights=torch.tensor([.5,.25,.25],device='cuda')
    for epoch in range(epochs):
        net.train()
        for ix in torch.randperm(len(w),device='cuda').split(4096):
            delta=2*torch.tanh(net(x[ix]));p=torch.softmax(lp[ix]+delta,dim=1)
            prob=torch.stack([1-p[:,0],p[:,1]+p[:,3],p[:,2]+p[:,3]],dim=1)
            loss=((((prob-y[ix])**2)*loss_weights).sum(dim=1)*w[ix]).mean()+penalty*delta.square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);optimizer.step()
    return dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},mean=mean,std=std,seed=seed,epochs=epochs,penalty=penalty)


def predict(feature,prior,model):
    x=torch.tensor(np.clip((feature-model['mean'])/model['std'],-8,8),device='cuda',dtype=torch.float32)
    lp=torch.tensor(np.log(np.clip(prior,1e-6,1)),device='cuda',dtype=torch.float32);net=network(x.shape[1]).cuda();net.load_state_dict(model['state_dict']);net.eval()
    with torch.inference_mode():return torch.softmax(lp+2*torch.tanh(net(x)),dim=1).cpu().numpy()
