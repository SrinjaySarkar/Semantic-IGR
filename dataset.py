import torch
import torch.utils as utils
import torch.utils.data as data
import numpy as np
from scipy import spatial
import os
import json


def get_filenames(base_dir,split,ext="",format="npy"):
    npyfiles=[]
    l=0
    for dataset in split:
        # print(dataset)
        for class_name in split[dataset]:
            # print(class_name)
            for instance_name in split[dataset][class_name]:
                # print(instance_name)
                j=0
                for shape in split[dataset][class_name][instance_name]:
                    instance_filename=os.path.join(base_dir,class_name,instance_name,shape+"{0}.{1}".format(ext,format))
                    # print(instance_filename)
                    assert(os.path.exists(instance_filename))
                    l+=1
                    j+=1
                    npyfiles.append(instance_filename)
    return(npyfiles)


# train_split_file="/vinai/sskar/unsup_implicit/dfaust/train_all.json"
# with open(train_split_file, "r") as f:
#      train_split=json.load(f)


class faust_dataset(utils.data.Dataset):
    def __init__(self,dataset_path,split,points_batch=16384,d_in=3,with_gt=False,with_normals=True):
        base_dir=os.path.abspath(dataset_path)
        self.npy_files=get_filenames(base_dir,split)
        # print(self.npy_files)
        self.points_batch=points_batch
        self.with_normals=with_normals
        self.d_in=d_in

        if with_gt:
            self.scan_files=get_instance_filenames("/vinai/sskar/unsup_implicit/dfaust/scans",split,'',"ply")
            self.shapenames=[x.split("/")[-1].split(".ply")[0] for x in self.scan_files]
    
    def load_points(self,idx):
        pc=np.load(self.npy_files[idx])
        return (pc)
    
    def get_info(self,idx):
        shape_name,pose,tag=self.npyfiles[idx].split("/")[-3:]
        return (shape_name,pose,tag[:tag.find(".npy")])
    
    def __getitem__(self,idx):
        point_cloud=torch.from_numpy(self.load_points(idx)).float()
        random_idx=torch.randperm(point_cloud.shape[0])[:self.points_batch]
        point_cloud=torch.index_select(point_cloud,0,random_idx)
        # print("#############333",point_cloud[:,:self.d_in].shape)
        
        if self.with_normals:
            normals=point_cloud[:,-self.d_in:]
        else:
            normals=torch.empty(0)
        return (point_cloud[:,:self.d_in],normals,idx)
    
    def __len__(self):
        length=len(self.npy_files)
        return (length)
        
# for p in np.array_split(data,100,axis=0):
#     d=ptree.query(p,50+1)
#     sigma_set.append(d[0][:,-1])
# sigmas=np.concatenate(sigma_set)



