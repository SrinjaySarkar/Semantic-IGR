import os
import sys
from tqdm import tqdm
from collections import OrderedDict
from tabnanny import filename_only
import open3d as o3d
from time import time
import numpy as np
import torch
import torch.optim as optim
import torch.utils as utils
from torch import autograd
import json
from model import implicit_network2,sampler,grads,init_network
from scipy.spatial import cKDTree
from shapenet_dataset import shapenet_v0,single_shape_dataset
from render_mesh import get_mesh
from scipy import spatial
from scipy.spatial import cKDTree


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_dim=3
noise=0.3
train=True
eval=True
save_sub_pc=True
save=False
resolution=128

#loading single point cloud
def load_points(file_path,n_points):
    pcd=o3d.io.read_point_cloud(file_path)
    pts=np.asarray(pcd.points)

    #stanadardize
    size=pts.max(axis=0)-pts.min(axis=0)
    pts=2*pts/size.max()
    pts-=(pts.max(axis=0)+pts.min(axis=0))/2

    #center scaling
    center=np.mean(pts,axis=0)
    pts=pts-np.expand_dims(center,axis=0)
    choice=np.random.choice(pts.shape[0],n_points,replace=False)
    pts=pts[choice,:]
    return (pts)


cls_choice="Airplane"#"Bunny"
bunny_path="/vinai/sskar/unsup_implicit/bunny_pc.ply"
airplane_path="/vinai/sskar/unsup_implicit/v0_plane1.ply"

if cls_choice == "Airplane":
    point_cloud=load_points(file_path=airplane_path,n_points=2048)
else:
    point_cloud=load_points(file_path=bunny_path,n_points=2048)

if save:
    name="std"+cls_choice
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(name+"ply",pcd)


#setting dynamic batch size
def bs(iter):
    bs_dict=[{"epoch":10,"batch_size":32},{"epoch":20,"batch_size":64},{"epoch":30,"batch_size":128},{"epoch":40,"batch_size":256},
            {"epoch":50,"batch_size":512},{"epoch":100,"batch_size":1024}]
    
    for s in bs_dict:
        if iter<s["epoch"]:
            # print(s["batch_size"])
            return (s["batch_size"])
    return (2048)

#model
net=implicit_network2(input_dim=input_dim)
net=net.to(device)
net=init_network(net)

#dataloader and optimizer 
optimizer=torch.optim.Adam(net.parameters(),lr=0.0001)
ssd=single_shape_dataset(point_cloud=point_cloud)
data_loader=torch.utils.data.DataLoader(ssd,batch_size=32,shuffle=True)


if train:
    print("############  Training")
    for epoch in range(0,n_epochs):

        batch_size=bs(epoch)
        if batch_size!=32:
            data_loader=utils.data.DataLoader(ssd,batch_size=batch_size,shuffle=True)
        else:
            pass
        total_loss=0
        total_count=0
        net.train()
        for batch_idx,batch in enumerate(data_loader):
            pts,rad=batch[0],batch[1]
            #saving part of the point clouds
            if save_sub_pc :
                pcd=o3d.geometry.PointCloud()
                pcd.points=o3d.utility.Vector3dVector(pts.numpy())
                o3d.io.write_point_cloud("/vinai/sskar/unsup_implicit/sample_pc/"+str(batch_idx)+".ply",pcd)

            # print(pts.shape)
            # print(rad.shape)

            batch_size=pts.shape[0]
            net.zero_grad()
            offsusrface=pts+torch.normal(0,1,pts.shape)*rad.float().unsqueeze(1)
            offsusrafce=torch.Tensor(offsusrface)

            #save offsurface
            if save_sub_pc:
                pcd=o3d.geometry.PointCloud()
                pcd.points=o3d.utility.Vector3dVector(offsurface.numpy())
                o3d.io.write_point_cloud("/vinai/sskar/unsup_implicit/sample_pc/"+str(batch_idx)+"_offsurface1.ply",pcd)
            
            #save uniform
            uniform=3+torch.rand_like(fake)-1.5
            if save_sub_pc:
                pcd=o3d.geometry.PointCloud()
                pcd.points=o3d.utility.Vector3dVector(uniform.numpy())
                o3d.io.write_point_cloud("/vinai/sskar/unsup_implicit/sample_pc/"+str(batch_idx)+"_uniform.ply",pcd)
            
            #save fake
            fake=torch.cat((offsurface,uniform),axis=0)
            fake.requires_grad_()
            if save_sub_pc:
                pcd=o3d.geometry.PointCloud()
                pcd.points=o3d.utility.Vector3dVector(fake.detach().numpy())
                o3d.io.write_point_cloud("/vinai/sskar/unsup_implicit/sample_pc/"+str(batch_idx)+"_offsurface2.ply",pcd)
            
            pts=pts.to(device)
            model_op=net(pts)
            loss_pointcloud=(model_op.abs()).mean()
            fake_op=net(fake)
            fake_op=fake_op.to(device)
            fake_grad=grads(fake,fake_op)[0]
            eikonal_loss=((fake_grad.norm(2,dim=1)-1)**2).mean()
            loss=loss_pointcloud + (0.1*eikonal_loss)

            total_loss+=loss.item()
            total_loss=total_loss/total_count
            loss.backward()
            optimizer.step()
        
        print(epoch,loss)


    print("############  End of training")
    print("############  Saving model")

    torch.save(net.state_dict(),'{}.pth'.format("single_shape_weights"))
    torch.save(optimizer.state_dict(),'{}.pth'.format("single_shape_opt"))

    print("############  Saved model")



if eval:
    net.load_state_dict("/vinai/sskar/unsup_implicit/single_shape_Wights.pth")
    net.eval()

    x=np.linspace(-1.2,1.2,resolution)
    y=x
    z=x
    X,Y,Z=np.meshgrid(x,y,z)
    X,Y,Z=X.reshepe(-1),Y.reshepe(-1),Z.reshepe(-1)
    
    pts=np.stack((X,Y,Z),axis=1)
    pts=pts.reshape(resolution,-1,3)

    vals=[]
    for point in tqdm.tqdm(pts):
        v=net(torch.Tensor(point).to(device))
        v=v.reshape(-1).detach().cpu().numpy()
        vals.append(v)
    pts=pts.reshape((-1,3))
    val=np.concatenate(vals)

    volume=val.reshape(resolution,resolution,resolution)
    verts,faces,normals,values=measure.marching_cubes(volume,0.0,spacing=(1.0,1.0,1.0))
    
    #save rendered mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)
    
    os.makedirs('output',exist_ok=True)
    o3d.io.write_triangle_mesh("new_mesh.ply",mesh)

