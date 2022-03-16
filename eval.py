import os
import sys
import json
from collections import OrderedDict
import torch
import torch.utils as utils
import torch.optim as optim
from dataset import faust_dataset
from model import implicit_network,sampler,grads
from render_mesh import get_mesh
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def adjust_learning_rate(initial_lr,optimizer,iter):
    adjust_lr_every=400             
    lr=initial_lr*((0.1)**(iter // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"]=lr

def opt_lat(point_cloud,normals,n_iterations,network,lr):
    latent_size=256
    global_sigma=1.8
    local_sigma=0.01
    sampler_model=sampler(global_sigma,local_sigma)
    latent_lambda=1e-3
    normals_lambda=1.0
    grad_lambda=0.1
    sampler_model=sampler(global_sigma,local_sigma)

    n_points,dim=point_cloud.shape
    latent=torch.ones(latent_size).normal_(0,1/latent_size).to(device)
    latent.requires_grad=True
    optimizer=optim.Adam([latent],lr=lr)

    for i in range(n_iterations):
        sample=sampler_model.get_points(point_cloud.unsqueeze(0)).squeeze()#(on+off) surface (28125,3)

        latent_all=latent.expand(n_points,-1)#(n_points,256)
        surface_points=torch.cat([latent_all,point_cloud],dim=1)#(on_surface_points,259)
        surface_points=surface_points.to(device)
        surface_points.requires_grad_()

        sample_latent_all=latent.expand(sample.shape[0],-1)
        offsurface_points=torch.cat([sample_latent_all,sample],dim=1)#(off+on_surface_points,259)
        offsurface_points=offsurface_points.to(device)
        offsurface_points.requires_grad_()

        surface_preds=network(surface_points)
        off_surface_preds=network(offsurface_points)

        surface_grad=grads(surface_points,surface_preds)[0][:,-3:]
        offsurface_grad=grads(offsurface_points,off_surface_preds)[0][:,-3:]

        surface_loss=torch.abs(surface_preds).mean()
        grad_loss=torch.mean((offsurface_grad.norm(2,dim=-1)-1).pow(2))
        normals_loss=((surface_grad-normals).abs()).norm(2,dim=1).mean()
        latent_loss=latent.abs().mean()
        
        loss=surface_loss+(latent_lambda*latent_loss)+(normals_lambda*normals_loss)+(grad_lambda*grad_loss)
        adjust_learning_rate(lr,optimizer,i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('latent loss iter {0}:{1}'.format(i, loss.item()))
        
    return (latent.unsqueeze(0))


d_in=3
n_iter=800
latent_size=256
latent_lambda=1e-3
grad_lambda=0.1
normals_lambda=1.0
with_normals=normals_lambda>0
dims=[512,512,512,512,512,512,512,512]
network=implicit_network(d_in=d_in+latent_size,dims=dims,skip_in=[4],geometric_init=True,radius_init=1,beta=100).to(device)

#load weights
x=torch.load("/vinai/sskar/unsup_implicit/1200.pth")
new_state_dict=OrderedDict()
print("##")
for k,v in x["model_state_dict"].items():
    name=k
    name=name[7:]
    new_state_dict[name]=v
network.load_state_dict(new_state_dict)



split_file="/vinai/sskar/unsup_implicit/dfaust/test_all.json"
with open(split_file,"r") as f:
    test_split=json.load(f)
print(test_split)

ds=faust_dataset(dataset_path="/vinai/sskar/unsup_implicit/dfaust/preprocessed/",split=test_split,with_normals=True)
total_files=len(ds)
counter=0
print("total test files: {0}".format(total_files))
test_datalaoder=utils.data.DataLoader(ds,batch_size=1,shuffle=True,num_workers=1,drop_last=False)


for batch_idx,(point_cloud,normals,idxs) in enumerate(test_datalaoder):
    if batch_idx == 0 :
        point_cloud=point_cloud.to(device).squeeze()
        print("pointcloud",point_cloud.shape)
        normals=normals.to(device).squeeze()
        # print("normals",normals.shape)
        print(counter)
        counter+=1
        network.train()
        latent=opt_lat(point_cloud,normals,n_iter,network,lr=6e-3)
        # latent=torch.ones(latent_size).normal_(0,1/latent_size).to(device)
        all_latent=latent.repeat(point_cloud.shape[0],1)#(n_points,256)
        # print(latent.shape)
        # print(all_latent.shape)
        points=torch.cat([all_latent,point_cloud],dim=-1)
        # print(points.shape)
        with torch.no_grad():
            network=network.eval()
            get_mesh(with_points=True,points=points,model=network,latent=latent,epoch=1,resolution=32,mc_value=0,uniform_grid=False,
                    verbose=False,save_ply=True,save_html=False,connected=False)
    else:
        break
