# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:34:22 2020

@author: Pranjal Vithlani
"""

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset




class UCF101Dataset(Dataset):
    """action recognition dataset dataset."""

    def __init__(self, csv_file, root_dir, transform=None, nframes_per_clip = 8):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.clip_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.nframes = nframes_per_clip

    def __len__(self):
        return len(self.clip_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.clip_frame.iloc[idx, 1],
                                self.clip_frame.iloc[idx, 0])
        
        label = self.clip_frame.iloc[idx, 2] 
        
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
            
        t_images = image.clone()
        for i in range(len(self.nframes)):
            #replacing the frame no. with "i"
            if i < 10:
                img_name[-6:-4] = "0"+str(i)  
            else:
                img_name[-6:-4] = str(i)
            image = io.imread(img_name)
    
            if self.transform:
                image = self.transform(image)
            
            t_images = torch.cat(t_images, image)
        
        sample = {'images':t_images, 'label':label}

        return sample