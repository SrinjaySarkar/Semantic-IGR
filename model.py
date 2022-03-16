import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

class implicit_network(torch.nn.Module):
    def __init__(self,d_in,dims,skip_in=(),geometric_init=True,radius_init=1,beta=100):
        super().__init__()
        dims=[d_in]+dims+[1]
        self.n_layers=len(dims)
        self.skip_in=skip_in
        for layer in range(0,self.n_layers-1):
            if layer+1 in skip_in:
                # print(dims[layer+1],d_in)
                out_dim=dims[layer+1]-d_in
            else:
                out_dim=dims[layer+1]
            lin=torch.nn.Linear(dims[layer],out_dim)
            if geometric_init:
                if layer==self.n_layers-2:
                    torch.nn.init.normal_(lin.weight,mean=np.sqrt(np.pi)/np.sqrt(dims[layer]),std=0.00001)
                    torch.nn.init.constant_(lin.bias,-radius_init)
                else:
                    torch.nn.init.constant_(lin.bias,0.0)
                    torch.nn.init.normal_(lin.weight,0.0,np.sqrt(2)/np.sqrt(out_dim))
            setattr(self,"lin"+str(layer),lin)
        if beta>0:
            self.ac_fn=torch.nn.Softplus(beta)
        else:
            self.ac_fn=torch.nn.ReLU()
    
    def forward(self,p):
        x=p
        for layer in range(0,self.n_layers-1):
            lin=getattr(self,"lin"+str(layer))
            if layer in self.skip_in:
                x=torch.cat([x,p],dim=-1)/np.sqrt(2)
            x=lin(x)
            if (layer<self.n_layers-2):
                x=self.ac_fn(x)
        return (x)


"""sampler is another way of getting a tuple of [on surface points,off surface points].this is used in conv occ points to."""
class sampler():
    def __init__(self,global_sigma,local_sigma=0.01):
        self.local_sigma=local_sigma
        self.global_sigma=global_sigma
    
    def get_points(self,pc,local_sigma=None):
        batch_size,sample_size,dim=pc.shape
        if local_sigma is not None:
            sample_local=pc+(torch.randn_like(pc)*local_sigma.unsqueeze(-1))
        else:
            sample_local=pc+(torch.randn_like(pc)*self.local_sigma)
        sample_global=(torch.rand(batch_size,sample_size//8,dim,device=pc.device)*(self.global_sigma*2))-self.global_sigma        
        
        sample=torch.cat([sample_local,sample_global],dim=1)
        return (sample)

def grads(ips,ops):
    d_points=torch.ones_like(ops,requires_grad=False,device=ops.device)
    points_grad=grad(outputs=ops,inputs=ips,grad_outputs=d_points,create_graph=True,retain_graph=True,only_inputs=True)
    return(points_grad)



class implicit_network2(torch.nn.Module):
    def __init__(self,input_dim):
        super(implicit_network,self).__init__()
        self.l1=torch.nn.Linear(input_dim,512)
        self.l2=torch.nn.Linear(512,512)
        self.l3=torch.nn.Linear(512,512)
        self.l4=torch.nn.Linear(512,512-input_dim)
        self.l5=torch.nn.Linear(512,512)
        self.l6=torch.nn.Linear(512,512)
        self.l7=torch.nn.Linear(512,512)
        self.l8=torch.nn.Linear(512,512)
        self.l9=torch.nn.Linear(512,512)
        self.l10=torch.nn.Linear(512,512)
        self.l11=torch.nn.Linear(512,512)
        self.l12=torch.nn.Linear(512,512)
        self.l13=torch.nn.Linear(512,512)
        self.l14=torch.nn.Linear(512,512)
        self.l15=torch.nn.Linear(512,512)
        self.l_out=torch.nn.Linear(512,1)
    
    def forward(self,x):
        h=F.softplus(self.l1(x),beta=100)
        h=F.softplus(self.l2(x),beta=100)
        h=F.softplus(self.l4(x),beta=100)
        h=F.softplus(self.l4(x),beta=100)
        h=F.softplus(self.l5(x),beta=100)
        h=F.softplus(self.l6(x),beta=100)
        h=F.softplus(self.l7(x),beta=100)
        h=F.softplus(self.l8(x),beta=100)
        h=F.softplus(self.l9(x),beta=100)
        h=F.softplus(self.l10(x),beta=100)
        h=F.softplus(self.l11(x),beta=100)
        h=F.softplus(self.l12(x),beta=100)
        h=F.softplus(self.l13(x),beta=100)
        h=F.softplus(self.l14(x),beta=100)
        h=F.softplus(self.l15(x),beta=100)
        h=self.l_out(h)
        return (h)
    

def init_network(network):
    for k,v in network.named.parameters():
        if "weight" in k:
            std=np.sqrt(2)/np.sqrt(v.shape[0])
            torch.nn.init.normal_(v,0.0,std)
        if "bias" in k:
            torch.nn.init.constant_(v,0)
        if k=="l_out.weight":
            std=np.sqrt(np.pi)/np.sqrt(v.shape[1])  
            torch.nn.init.constant_(v,std)
        if k=="l_out.bias":
            torch.nn.init.constant_(v,-1)
    return (network)