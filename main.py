# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:06:50 2020

@author: Pranjal Vithlani
"""


import cv2
import numpy as np
import argparse
import os
import shutil
import time
from datetime import datetime
from params import argparams
from dataUtils import UCF101Dataset
import models
import matplotlib.pyplot as plt
plt.ion()   # interactive mode

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#import torchvision.models as models



best_prec1 = 0

def main(argparams):
    global args, best_prec1
    
    num_classes = 101
    args = argparams
    # args.arch = 'r2plus1d_18'
    # model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
    # #print(model_names)
    
    # args.arch = 'resnet18'
    # # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()
    
    args.arch = 'r2plus1d_18'
    
    # save checkpoints and plots
    now = datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    save_exps = '{}_{}'.format('exp', timestamp)
    save_exps = './runs/'+save_exps + '/'
    if not os.path.exists(save_exps):
        os.makedirs(save_exps)
    
    model = models.r2plus1d_18(args.pretrained)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = torch.nn.DataParallel(model).cuda()
    
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = './data/trainlist01_frames8.csv'
    valdir = './data/testlist01_frames8.csv'
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_img_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    transform_img_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    
    train_data = UCF101Dataset(traindir, args.data, transform=transform_img_train)
    train_loader = torch.utils.data.DataLoader(train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_data = UCF101Dataset(valdir, args.data, transform=transform_img_val)
    val_loader = torch.utils.data.DataLoader(val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
        visualize_model(model, val_loader, num_images=6)
        return

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_prec1, train_prec5, train_loss = train(train_loader, model, criterion, 
                                         optimizer, epoch)
        train_acc_history.append(train_prec1)
        train_loss_history.append(train_loss)
        
        # evaluate on validation set
        prec1, prec5, val_loss = validate(val_loader, model, criterion)
        val_acc_history.append(prec1)
        val_loss_history.append(val_loss)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, save_exps)
        plot_results(train_acc_history, val_acc_history, save_exps, 
                     loss_plot = False)
        plot_results(train_loss_history, val_loss_history, save_exps, 
                     loss_plot = True)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample_batched in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = sample_batched['images']
        target = sample_batched['label']
        
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            
    return top1.avg, top5.avg, loss.data.item()

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, sample_batched in enumerate(val_loader):
            
            input = sample_batched['images']
            target = sample_batched['label']
            
            target = target.cuda(non_blocking=True)
            input_var = input
            target_var = target
    
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
    
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, loss.data.item()


def save_checkpoint(state, is_best, save_exps, filename='checkpoint.pth.tar'):
    torch.save(state, save_exps+filename)
    if is_best:
        shutil.copyfile(save_exps+filename, save_exps+'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def plot_results(train_history, val_history, save_exps, loss_plot):
    if loss_plot:
        plt.title("Loss vs. Number of Training Epochs")
        plt.ylabel("Loss")
    else:
        plt.title("Accuracy vs. Number of Training Epochs")
        plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.plot(range(1,args.epochs+1),train_history,label="Train")
    plt.plot(range(1,args.epochs+1),val_history,label="Validation")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, args.epochs+1, 1.0))
    plt.legend()
    #plt.show()
    if loss_plot:
        plt.savefig(save_exps+'loss_plot.jpg')
    else:
        plt.savefig(save_exps+'accuracy_plot.jpg')
    return

def visualize_model(model, val_loader, num_images=6):
    was_training = model.training
    model.eval()
    # images_so_far = 0
    # fig = plt.figure()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    filename = open('./data/classInd.txt','r')
    class_names = filename.readlines()
    class_names = [i.split(" ")[1] for i in class_names]

    with torch.no_grad():
        for i, sample_batched in enumerate(val_loader):
            images_so_far = 0
            fig = plt.figure()
            
            inputs = sample_batched['images']
            labels = sample_batched['label']
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    break #return
        model.train(mode=was_training)
        
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 3, 0))
    inp = inp[4]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated
    
        
if __name__ == '__main__':
    main(argparams)