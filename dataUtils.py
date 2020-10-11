# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:34:22 2020

@author: Pranjal Vithlani
"""

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

import numpy as np
import cv2
import os
import torch
import pandas as pd
from PIL import Image
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
            nframes_per_clip (int, optional): number of frames to be extracted per clip
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
        
        img_name_use = img_name[:-6] + "00" + img_name[-4:]
        image = Image.open(img_name_use)
        if self.transform:
            image = self.transform(image)
        
        imageR = image[0].clone().unsqueeze(0)
        imageG = image[1].clone().unsqueeze(0)
        imageB = image[2].clone().unsqueeze(0)
        
        for i in range(1,self.nframes):
            #replacing the frame no. with "i"
            if i < 10:
                img_name_use = img_name[:-6] + "0"+str(i) + img_name[-4:]
            else:
                img_name_use = img_name[:-6] + str(i) + img_name[-4:]
            
            image = Image.open(img_name_use)
    
            if self.transform:
                image = self.transform(image)
            
            imageR = torch.cat((imageR, image[0].unsqueeze(0)))
            imageG = torch.cat((imageG, image[1].unsqueeze(0)))
            imageB = torch.cat((imageB, image[2].unsqueeze(0)))
            
        t_combine = []
        t_combine.append(imageR.unsqueeze(0))
        t_combine.append(imageG.unsqueeze(0))
        t_combine.append(imageB.unsqueeze(0))
        t_images = torch.cat(t_combine,0)
        
        sample = {'images':t_images, 'label':label}

        return sample
    
    
class UCF101DatasetPred(Dataset):
    """action recognition dataset dataset."""

    def __init__(self, video_path, transform=None, nframes_per_clip = 8):
        """
        Args:
            video_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            nframes_per_clip (int, optional): number of frames to be extracted per clip
        """
        self.video_path = video_path
        self.transform = transform
        self.nframes = nframes_per_clip

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # extract frames code
        vidcap = cv2.VideoCapture(self.video_path)
        success,image = vidcap.read()
        count = 0
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)-1
        multiplier = total_frames // self.nframes
        while success:
            frameId = int(round(vidcap.get(1)))
            if frameId % multiplier == 0:
                image = Image.fromarray(np.uint8(image)).convert('RGB')
                if self.transform:
                        image = self.transform(image)
                if count == 0:
                    imageR = image[0].clone().unsqueeze(0)
                    imageG = image[1].clone().unsqueeze(0)
                    imageB = image[2].clone().unsqueeze(0)
                else:                    
                    imageR = torch.cat((imageR, image[0].unsqueeze(0)))
                    imageG = torch.cat((imageG, image[1].unsqueeze(0)))
                    imageB = torch.cat((imageB, image[2].unsqueeze(0)))
                    
                count+=1
                
            if count == self.nframes:
                break
            success,image = vidcap.read()
            
        vidcap.release()
        
        t_combine = []
        t_combine.append(imageR.unsqueeze(0))
        t_combine.append(imageG.unsqueeze(0))
        t_combine.append(imageB.unsqueeze(0))
        t_images = torch.cat(t_combine,0)
        
        sample = {'images':t_images}

        return sample