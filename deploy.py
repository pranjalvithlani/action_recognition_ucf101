# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:38:15 2020

@author: Pranjal Vithlani
"""



import cv2
import numpy as np
import os

from params import argparams
from dataUtils import UCF101DatasetPred
import models

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#import torchvision.models as models


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)



global args

num_classes = 101
args = argparams

args.arch = 'r2plus1d_18'
args.batch_size = 1
args.evaluate = True

model = models.r2plus1d_18(args.pretrained)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = torch.nn.DataParallel(model).cuda()

filename = open('./data/classInd.txt','r')
class_names = filename.readlines()
class_names = [i.split(" ")[1] for i in class_names]

# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True
model.eval()

def model_predict(file_path):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    
    transform_img_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    

    val_data = UCF101DatasetPred(file_path, transform=transform_img_val)
    val_loader = torch.utils.data.DataLoader(val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
        
    
    device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    

    with torch.no_grad():
        for i, sample_batched in enumerate(val_loader):
            
            inputs = sample_batched['images']
            
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # print("ans", class_names[preds[0]])
        
    return class_names[preds[0]]

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)

        return preds
    return None
       
if __name__ == '__main__':
    app.run(debug=True)