from torch.utils.data import Dataset
import os
import h5py
from torchvision import transforms
import torch
import random

class MyDataset(Dataset):
    def __init__(self,root_dir,splits_path,selected_number,mode,samples,transform=None,task="digit"):
        self.task = task
        self.root_dir = root_dir
        self.splits_path = splits_path
        self.selected_number = selected_number
        self.mode = mode
        self.transform = transform
        self.path = os.path.join(self.root_dir,self.splits_path,"AlexNet_digit_{}_{}.txt".format(self.selected_number,self.mode))
        self.path_list = []
        with open(self.path,"r") as f:
            for path in f:
                self.path_list.append(path.rstrip("\n"))
        self.path_list = random.sample(self.path_list,samples)
        self.task = task

    def __getitem__(self, idx):

        img_path = self.path_list[idx]
        img_file = h5py.File(img_path,'r')
        img = img_file["data"][0]
        img = torch.tensor(img,dtype=torch.float32)
        #standardization
        img_min = img.min()
        img_max = img.max()
        img_scale = img_max-img_min
        img_scaled = (img+img_scale)/img_scale
        if self.task=="digit":
            label = img_file["label"][0][0]
        else:
            #task=gender
            label = img_file["label"][0][1]
        label = torch.tensor(label,dtype=torch.int64)
        if self.transform:
            img_scaled = self.transform(img_scaled)

        return img_scaled,label

    def __len__(self):
        return len(self.path_list)






