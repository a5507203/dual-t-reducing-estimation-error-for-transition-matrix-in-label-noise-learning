import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import yuyao.models as models
import argparse, sys
import numpy as np
from yuyao.data.data_loader import DataLoader_noise
import datetime
import shutil
from loss import loss_forget
import types

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=mentor_args.alpha_plan[epoch]
        param_group['betas']=(mentor_args.beta1_plan[epoch], 0.999) # Only change beta1
        

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2):
 
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 

    for i, (images, labels, _) in enumerate(train_loader):
        if i>mentor_args.num_iter_per_epoch:
            break
      
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        logits1=model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec1

        logits2 = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2+=1
        train_correct2+=prec2
        loss_1 = loss_forget(logits1, labels, mentor_args.rate_schedule[epoch])
        loss_2 = loss_forget(logits2, labels, mentor_args.rate_schedule[epoch])                          
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i+1) % mentor_args.print_freq == 0:
            print ('Epoch=',epoch+1,'Training Accuracy1:',prec1.item(),'Training Accuracy2',prec2.item())

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2


# Evaluate the Model
def evaluate(test_loader, model1, model2):

    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2


mentor_args = types.SimpleNamespace()

def run_mentoring(train_dataset, test_dataset, noise_rate, input_channel, pretrained, arch="CNN", num_classes = 10, n_epoch=200):
    global mentor_args

    # Hyper Parameters
    mentor_args.rate_schedule = None
    mentor_args.lr = 0.001
    mentor_args.num_gradual = 10
    mentor_args.exponent = 1
    mentor_args.n_epoch = 200
    mentor_args.print_freq = 100
    mentor_args.num_workers = 0
    mentor_args.num_iter_per_epoch = 400
    mentor_args.num_classes = num_classes
    mentor_args.epoch_decay_start = 80
    mentor_args.pretrained = pretrained
    mentor_args.arch = arch
    batch_size = 128
    learning_rate = mentor_args.lr 

    # Adjust learning rate and betas for Adam Optimizer
    mom1 = 0.9
    mom2 = 0.1
    mentor_args.alpha_plan = [learning_rate] * mentor_args.n_epoch
    mentor_args.beta1_plan = [mom1] * mentor_args.n_epoch

    for i in range(mentor_args.epoch_decay_start, mentor_args.n_epoch):
        mentor_args.alpha_plan[i] = float(mentor_args.n_epoch - i) / (mentor_args.n_epoch - mentor_args.epoch_decay_start) * learning_rate
        mentor_args.beta1_plan[i] = mom2

    train_dataset.transforms=transforms.ToTensor()
    test_dataset.transforms=transforms.ToTensor()
    mentor_args.n_epoch = n_epoch
    mentor_args.noise_rate = noise_rate
    mentor_args.forget_rate = noise_rate
    # define drop rate schedule
    mentor_args.rate_schedule = np.ones(mentor_args.n_epoch)*mentor_args.forget_rate 
    mentor_args.rate_schedule[:mentor_args.num_gradual] = np.linspace(0, mentor_args.forget_rate **mentor_args.exponent, mentor_args.num_gradual)
    
    train_loader = DataLoader_noise(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=mentor_args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = DataLoader_noise(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=mentor_args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    cnn1 = models.__dict__[mentor_args.arch](input_channel=input_channel, num_classes=mentor_args.num_classes, pretrained = mentor_args.pretrained)
    cnn1.cuda()
  
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    
    cnn2 = models.__dict__[mentor_args.arch](input_channel=input_channel, num_classes=mentor_args.num_classes, pretrained = mentor_args.pretrained)
    cnn2.cuda()

    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

    epoch=0
    train_acc1=0
    train_acc2=0
    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%' % (epoch+1, mentor_args.n_epoch, len(test_dataset), test_acc1, test_acc2))
    test_acc  = 0
    # training
    for epoch in range(0, mentor_args.n_epoch):
        # train models
        train_loader.train()
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2=train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2)
        # evaluate models
        test_loader.eval()
        test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
    
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%' % (epoch+1, mentor_args.n_epoch, len(test_dataset), test_acc1, test_acc2))
        if mentor_args.n_epoch-epoch<=10:
            print(epoch)
            test_acc = test_acc+test_acc1+test_acc2
            
    test_acc = test_acc/20
    return test_acc