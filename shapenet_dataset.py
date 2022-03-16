import os 
import numpy as np
import torch
import json
import torch.utils as utils
from scipy import spatial


def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def pc_normalize(pc):
    # centroid=np.mean(pc,axis=0)
    # pc=pc-centroid
    # m=np.max(np.sqrt(np.sum(pc**2,axis=1)))
    # pc=pc/m
    return (pc)

class shapenet_v0(utils.data.Dataset):
    def __init__(self,root,npoints=2500,split='train',class_choice=None,normal_channel=False):
        self.npoints=npoints
        self.root=root
        self.catfile=os.path.join(self.root,'synsetoffset2category.txt')
        self.cat={}
        self.normal_channel=normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls=line.strip().split()
                self.cat[ls[0]]=ls[1]
        self.cat={k:v for k,v in self.cat.items()}
        self.classes_original=dict(zip(self.cat,range(len(self.cat))))

        if not class_choice is  None:
            self.cat={k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'),'r') as f:
            train_ids=set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'),'r') as f:
            val_ids=set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'),'r') as f:
            test_ids=set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item]=[]
            dir_point=os.path.join(self.root,self.cat[item])
            fns=sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split=='all':
                fns=[fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split=='train':
                fns=[fn for fn in fns if fn[0:-4] in train_ids]
            elif split=='val':
                fns=[fn for fn in fns if fn[0:-4] in val_ids]
            elif split=='test':
                fns=[fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token=(os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath=[]
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item,fn))

        self.classes={}
        for i in self.cat.keys():
            self.classes[i]=self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes={'Earphone':[16,17,18],'Motorbike':[30,31,32,33,34,35],'Rocket':[41,42,43],'Car':[8,9,10,11],
        'Laptop':[28,29],'Cap':[6,7],'Skateboard':[44,45,46],'Mug':[36,37],'Guitar':[19,20,21],'Bag':[4,5],'Lamp':[24,25,26,27],
        'Table':[47,48,49],'Airplane':[0,1,2,3],'Pistol':[38,39,40],'Chair':[12,13,14,15],'Knife':[22,23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

    def __getitem__(self,index):
        fn=self.datapath[index]
        # print(fn)
        cat=self.datapath[index][0]
        cls=self.classes[cat]
        cls=np.array([cls]).astype(np.int64)
        data=np.loadtxt(fn[1]).astype(np.float32)
        if not self.normal_channel:
            point_set=data[:,0:3]
        else:
            point_set=data[:,0:6]
        seg=data[:,-1].astype(np.int32)
        point_set[:,0:3]=pc_normalize(point_set[:,0:3])
        choice=np.random.choice(len(seg),self.npoints,replace=True)
        # resample
        point_set=point_set[choice,:]
        seg=seg[choice]        
        jittered_pc=point_set+np.random.normal(0,0.1,size=point_set.shape) # random jitter

        return (point_set,cls,seg,index)

    def __len__(self):
        return len(self.datapath)


# class_name="Airplane"
# bs=1
# mkdir_ifnotexists("/vinai/sskar/unsup_implicit/shapenet_v0/"+str(class_name))
# root="/vinai/sskar/TTA/shapenetcore_partanno_segmentation_benchmark_v0_normal"
# ds=shapenet_v0(root,npoints=2500,split='train',class_choice=class_name,normal_channel=False)
# train_loader=utils.data.DataLoader(ds,batch_size=bs,shuffle=True,num_workers=8,drop_last=False)
# for batch_idx,(point_cloud,cat_label,seg_label,idx) in enumerate(train_loader):
#     print(point_cloud.shape)
#     if batch_idx == 0:
#         np.save("v0_plane1.npy",point_cloud.squeeze().numpy())
#     elif batch_idx == 1:
#         np.save("v0_plane2.npy",point_cloud.squeeze().numpy())
#     else:
#         break
    # print(cat_label)
    # print(seg_label)

class single_shape_dataset(utils.data.Dataset):
    def __init__(self,point_cloud):
        tree=spatial.KDTree(point_cloud)
        dists,indices=tree.query(point_cloud,k=50+1)
        radius=dists[:,-1]
        self.point_cloud=point_cloud.astype(np.float32)
        self.radius=radius
    
    def __len__(self):
        length=len(self.point_cloud)
        return (length)
    
    def __getitem__(self,idx):
        pts=self.point_cloud[idx]
        radius=self.radius[idx]
        return (pts,radius)

