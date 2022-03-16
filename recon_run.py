import os
import sys
from collections import OrderedDict
sys.path.append("/vinai/sskar/unsup_implicit/dfaust")
# import open3d as o3d
from time import time
import numpy as np
import torch
import torch.optim as optim
import torch.utils as utils
import json
from model import implicit_network,sampler,grads
from scipy.spatial import cKDTree
from dataset import faust_dataset
from render_mesh import get_mesh
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print("new")
checkpoint_every_n=1000


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval,factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)), 5.0e-6)


def latent_size_reg(latent_vector,idx):
    latents=torch.index_select(latent_vector,0,idx)
    latent_loss=latent_vector.norm(dim=1).mean()
    return (latent_loss)

def get_lr_schedules(schedule_specs):
    schedules=[]
    for schedule_specs in schedule_specs:
        if schedule_specs["Type"]=="Step":
            schedules.append(StepLearningRateSchedule(schedule_specs["initial"],schedule_specs["interval"],schedule_specs["factor"]))
        else:
            print("no known learning rate schedulers")
    return (schedules)


def add_latent(points,indices,latent_vector):
    batch_size,n_points,dim=points.shape
    points=points.reshape(batch_size*n_points,dim)
    latent_inputs=torch.zeros(0).to(device)

    for ind in indices.numpy():
        latent_ind=latent_vector[ind]
        latent_repeat=latent_ind.expand(n_points,-1)
        latent_inputs=torch.cat([latent_inputs,latent_repeat],0)
    points=torch.cat([latent_inputs,points],1)
    return (points)

def adjust_learning_rate(lr_schedules,epoch,optimizer):
    for i,param_group in enumerate(optimizer.param_groups):
        param_group["lr"]=lr_schedules[i].get_learning_rate(epoch)

def save_ckpts(network,optimizer,latent_vector,epoch,model_subdir,opt_subdir,latent_subdir):
    torch.save({"epoch":epoch,"model_state_dict":network.state_dict()},os.path.join(model_subdir,str(epoch)+".pth"))
    torch.save({"epoch":epoch,"optimizer_state_dict":optimizer.state_dict()},os.path.join(opt_subdir,str(epoch)+".pth"))
    torch.save({"epoch":epoch,"latent_codes":latent_vector},os.path.join(opt_subdir,str(epoch)+".pth"))

   
#experiment settings
# torch.cuda.empty_cache()
is_continue=False
GPU_index=0
n_gpus=torch.cuda.device_count()
eval=False
model_subdir="/vinai/sskar/unsup_implicit/ckpts/model_parameters"
latent_subdir="/vinai/sskar/unsup_implicit/ckpts/latent_codes"
opt_subdir="/vinai/sskar/unsup_implicit/ckpts/opt_parameters"
############
#changes for shapenet dataset.
# input_file="/vinai-public-dataset/shapenet_corev2/ShapeNetCore.v2/02691156/10155655850468db78d106ce0a280f87/models/pointcloud_final.npz"#path to the npy point cloud file 
# data=utils.load_point_cloud_by_file_extension(input_file)
parallel=False
n_epochs=1#10000
batch_size=1
# if n_gpus>0:
#     batch_size=batch_size*n_gpus
global_sigma=1.8
local_sigma=0.01

sampler=sampler(global_sigma,local_sigma)

train_split_file='/vinai/sskar/unsup_implicit/dfaust/train_all.json'
with open(train_split_file,"r") as f:
    train_split=json.load(f) 

d_in=3
latent_size=256
latent_lambda=1e-3
grad_lambda=0.1
normals_lambda=1.0
with_normals=normals_lambda>0

ds=faust_dataset(dataset_path="/vinai/sskar/unsup_implicit/dfaust/preprocessed/",split=train_split,with_normals=with_normals)
n_scenes=len(ds)
print("length:",n_scenes)
train_loader=utils.data.DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=8,drop_last=False)
eval_loader=utils.data.DataLoader(ds,batch_size=1,shuffle=True,num_workers=0,drop_last=True)
dims=[512,512,512,512,512,512,512,512]
network=implicit_network(d_in=d_in+latent_size,dims=dims,skip_in=[4],geometric_init=True,radius_init=1,beta=100)

if parallel:
    network=torch.nn.DataParallel(network)
if torch.cuda.is_available():
    network=network.to(device)

#learning rate and weight decay
learning_rate_schedule=[{"Type":"Step","initial":0.005,"interval":500,"factor":0.5},{"Type":"Step","initial":0.001,"interval":500,
"factor":0.5}]
lr_schedules=get_lr_schedules(learning_rate_schedule)
weight_decay=0

#optimizer and training loop
start_epoch=0
latent_vector=torch.zeros(n_scenes,latent_size).to(device)
latent_vector.requires_grad_()

optimizer=optim.Adam([{"params":network.parameters(),"lr":lr_schedules[0].get_learning_rate(0),"weight_decay":weight_decay},
{"params":latent_vector,"lr":lr_schedules[1].get_learning_rate(0)}])

#load weight files
if is_continue:    
    data=torch.load(os.path.join("/vinai/sskar/unsup_implicit/ckpts/latent_codes","ckpt_latest",".pth"))
    latent_vector=data["latent_codes"].to(device)
    saved_model_state=torch.load(os.path.join("/vinai/sskar/unsup_implicit/ckpts/model_parameters","ckpt_latest",".pth"))
    network.load_state_dict(saved_model_state["model_state_dict"])
    data=torch.load(os.path.join("/vinai/sskar/unsup_implicit/ckpts/opt_parameters","ckpt_latest",".pth"))
    optimizer.load_state_dict(data["optmizer_state_dict"])
    start_epoch=saved_model_state["epoch"]


# print(x.keys())

#running the training loop
for epoch in range(start_epoch,n_epochs+1):
    if (epoch % checkpoint_every_n) == 0:
        save_ckpts(network,optimizer,latent_vector,epoch,model_subdir,opt_subdir,latent_subdir)
        with torch.no_grad():
            network.eval()
            points,normals,idxs=next(iter(train_loader))
            points=points.to(device)
            points=add_latent(points,idxs,latent_vector)
            latent=latent_vector[idxs[0]]
            torch.cuda.empty_cache()
            get_mesh(with_points=True,points=points,model=network,latent=latent,epoch=epoch,resolution=300,mc_value=0.0,uniform_grid=False,
                    verbose=False,save_ply=False,save_html=False,connected=False)

        # pc,local_sigma
    network.train()
    adjust_learning_rate(lr_schedules,epoch,optimizer)
    before_epoch=time()
    for batch_idx,(point_cloud,normals,idxs) in enumerate(train_loader):
        point_cloud=point_cloud.to(device)
        print(point_cloud.shape)
        if with_normals:
            normals=normals.to(device)
        off_surface_points=sampler.get_points(point_cloud,local_sigma=None)#this is a tuple of (on surface+off surface points)
        # print(off_surface_points.shape)
        # if batch_idx == 0:
        #     p1c=off_surface_points[0].cpu().numpy()
        #     pcd=o3d.geometry.PointCloud()
        #     pcd.points=o3d.utility.Vector3dVector(p1c)
        #     o3d.io.write_point_cloud("spc"+".ply",pcd)
        # else:
        #     break
        #add latent vector
        point_cloud=add_latent(point_cloud,idxs)
        print("on:",point_cloud.shape)#n,256+3
        off_surface_points=add_latent(off_surface_points,idxs)
        print("on+off:",off_surface_points.shape)#n,256+3

        #run through the network
        point_cloud.requires_grad_()
        off_surface_points.requires_grad_() 
        point_preds=network(point_cloud)
        print("on preds:",point_preds.shape)
        off_surface_preds=network(off_surface_points)
        print("on+off preds:",off_surface_preds.shape)
        
        #gradient of function wrt input
        points_grad=grads(point_cloud,point_preds)
        points_grad=points_grad[0][:,-3:]
        off_surface_grad=grads(off_surface_points,off_surface_preds)
        off_surface_grad=off_surface_grad[0][:,-3:]
        print(off_surface_grad.shape)

        #loss+eikonal_loss
        loss1=(point_preds.abs()).mean()#or take l1 norm 
        eikonal_loss=((off_surface_grad.norm(2,dim=-1)-1)**2).mean()
        loss=loss1+(grad_lambda*eikonal_loss)

        if with_normals:
            normals=normals.view(-1,3)
            normal_loss=((points_grad-normals).abs()).norm(2,dim=1).mean()
            loss=loss+(normals_lambda*normal_loss)
        else:
            normal_loss=torch.zeros(1)
        
        latent_loss=latent_size_reg(latent_vector,idxs.to(device))
        loss+=(latent_lambda*latent_loss)   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx%checkpoint_every_n) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tManifold loss: {:.6f}'
                          '\tGrad loss: {:.6f}\tLatent loss: {:.6f}\tNormals Loss: {:.6f}'.format(
                        epoch,batch_idx*batch_size,len(train_loader),100.*batch_idx/len(train_loader),
                               loss.item(),loss1.item(),eikonal_loss.item(),latent_loss.item(),normal_loss.item()))  


            after_epoch = time()
            print('epoch time {0}'.format(str(after_epoch-before_epoch)))
