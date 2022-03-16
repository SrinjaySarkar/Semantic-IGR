import plotly.graph_objs as go
import plotly.offline as offline 
import torch
import os
import trimesh
import numpy as np
from skimage import measure
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# print(device)

def tri_indices(simplices):
    return ([triplet[c] for triplet in simplices] for c in range(3))


def get_trace(points,caption=None,colorscale=None,color=None):
    if (type(points) == list):
        trace=[go.Scatter3d(x=p[0][:, 0],y=p[0][:, 1],z=p[0][:, 2],mode='markers',name=p[1],marker=dict(size=3,line=dict(width=2,),opacity=0.9,
        colorscale=colorscale,showscale=True,color=color,),text=caption) for p in points]
    else:
        trace=[go.Scatter3d(x=points[:,0].detach(),y=points[:,1].detach(),z=points[:,2].detach(),mode='markers',name='projection',
        marker=dict(size=3,line=dict(width=2,),opacity=0.9,colorscale=colorscale,showscale=True,color=color,),text=caption)]
    return trace


def get_grid(points,resolution):
    eps=0.1
    # print(points.shape)
    input_min=torch.min(points,dim=0)[0].squeeze().detach().cpu().numpy()
    # print(input_min)
    input_max=torch.max(points,dim=0)[0].squeeze().detach().cpu().numpy()
    # print(input_max)
    bounding_box=input_max-input_min
    # print(bounding_box)
    shortest_axis=np.argmin(bounding_box)#z for 1st eval sample in 500002 chicken_wings.
    if (shortest_axis==0):
        x=np.linspace(input_min[shortest_axis]-eps,input_max[shortest_axis] + eps, resolution)
        length=np.max(x)-np.min(x)
        y=np.arange(input_min[1]-eps,input_max[1]+length/(x.shape[0]-1)+eps,length/(x.shape[0]-1))
        z=np.arange(input_min[2]-eps,input_max[2]+length/(x.shape[0]-1)+eps,length/(x.shape[0]-1))
    elif (shortest_axis==1):
        y=np.linspace(input_min[shortest_axis]-eps,input_max[shortest_axis] + eps, resolution)
        length=np.max(y)-np.min(y)
        x=np.arange(input_min[0]-eps,input_max[0]+length/(y.shape[0]-1)+eps,length/(y.shape[0]-1))
        z=np.arange(input_min[2]-eps,input_max[2]+length/(y.shape[0]-1)+eps,length/(y.shape[0]-1))
    elif (shortest_axis==2):
        z=np.linspace(input_min[shortest_axis]-eps,input_max[shortest_axis] + eps, resolution)
        length=np.max(z)-np.min(z)
        y=np.arange(input_min[1]-eps,input_max[1]+length/(z.shape[0]-1)+eps,length/(z.shape[0]-1))
        x=np.arange(input_min[0]-eps,input_max[0]+length/(z.shape[0]-1)+eps,length/(z.shape[0]-1))
    # print(x.shape)
    # print(y.shape)
    # print(z.shape)
    xx,yy,zz=np.meshgrid(x,y,z)#grid of points
    grid_points=torch.tensor(np.vstack([xx.ravel(),yy.ravel(),zz.ravel()]).T,dtype=torch.float).to(device)
    # print(xx.shape,yy.shape,zz.shape)
    # print(grid_points.shape)
    d={"grid_points":grid_points,"shortest_axis":shortest_axis,"xyz":[x,y,z],"shortest_axis_index":shortest_axis}
    return (d)


def get_grid_uniform(resolution):
    x=np.linspace(-1.2,1.2,resolution)
    y=x
    z=x
    xx,yy,zz=np.meshgrid(x, y, z)
    grid_points=torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T,dtype=torch.float).to(device)
    return {"grid_points":grid_points,"shortest_axis_length":2.4,"xyz":[x,y,z],"shortest_axis_index": 0}


def get_surface(points,model,latent,resolution,mc_value,uniform_grid,verbose,save_ply,connected=False):
    trace=[]
    mesh_export=None
    if (uniform_grid):
        grid=get_grid_uniform(resolution)
    else:
        if points is not None:
            grid=get_grid(points[:,-3:],resolution)
        else:
            grid=get_grid(None,resolution)
    z=[]
    print("grid:",grid["grid_points"].shape)
    # print("points:",points.device)
    for i,pts in enumerate(torch.split(grid["grid_points"],100000,dim=0)):
        # print(i)
        # print('here {0}'.format(i/(grid['grid_points'].shape[0]//100000)*100))#divide the points into 50,000 point segments
        # print(pts.shape)
        if (not latent is None):
            pts=torch.cat([latent.expand(pts.shape[0],-1),pts],dim=1)
        op=model(pts)
        z.append(op.detach().cpu().numpy())
    z=np.concatenate(z,axis=0)
    # print(np.min(z))
    # print(np.max(z))
    print(not(np.min(z)>mc_value or np.max(z)<mc_value))
    # print("########################################3",z.shape)

    #this part
    if (not(np.min(z)>mc_value or np.max(z)<mc_value)):
        print("all good")
        z=z.astype(np.float64)
        verts,faces,normals,values=measure.marching_cubes(volume=z.reshape(grid["xyz"][1].shape[0],grid["xyz"][0].shape[0],
        grid["xyz"][2].shape[0]).transpose([1,0,2]),level=mc_value,spacing=(grid['xyz'][0][2]-grid['xyz'][0][1],
        grid['xyz'][0][2]-grid['xyz'][0][1],grid['xyz'][0][2]-grid['xyz'][0][1]))
        # print(z)
        verts=verts+np.array([grid["xyz"][0][0],grid["xyz"][1][0],grid["xyz"][2][0]])
        if save_ply:
            meshexport=trimesh.Trimesh(verts,faces,normals,vertex_colors=values)
        I,J,K=tri_indices(faces)
        trace.append(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=I,j=J,k=K,name="",color="orange",opacity=0.5))
    
    return {"mesh_trace":trace,"mesh_export":meshexport}


def get_mesh(model,epoch,resolution,mc_value,uniform_grid,verbose,save_ply,save_html,points,with_points,latent,connected):
    filename="/vinai/sskar/unsup_implicit/gen_meshes/single_shape/mesh"+str(epoch)
    if with_points:
        points_val=model(points)#model spits out a (n_point,1) vector.
        points_val=points_val.cpu()
        points=points.cpu()
        caption=["decoder:{0}".format(val.item()) for val in points_val.squeeze()]
        # print("caption:",caption)
        trace_points=get_trace(points[:,-3:],caption)
        # print(trace_points)

    #this part
    surface=get_surface(points,model,latent,resolution,mc_value,uniform_grid,verbose,save_ply,connected)
    trace_surface=surface["mesh_trace"]
    layout=go.Layout(title=go.layout.Title(text="50002_chicken_wings"),width=1200,height=1200,scene=dict(xaxis=dict(range=[-2, 2],autorange=False),
                    yaxis=dict(range=[-2, 2], autorange=False),zaxis=dict(range=[-2, 2], autorange=False),aspectratio=dict(x=1, y=1, z=1)))
    if (with_points):
        fig1=go.Figure(data=trace_points+trace_surface,layout=layout)
    else:
        fig1=go.Figure(data=trace_surface,layout=layout)
    if (save_html):
        offline.plot(fig1,filename=filename+'.html',auto_open=False)
    if (not surface['mesh_export'] is None):
        surface['mesh_export'].export(filename+'.ply','ply')
        print("saved mesh for epoch:",epoch)
    return surface['mesh_export']





