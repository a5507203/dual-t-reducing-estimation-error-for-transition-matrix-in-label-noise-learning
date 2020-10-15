
from __future__ import print_function
import os
import sys
import time
import math
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import types
from yuyao.data.data_loader import DataLoader_noise
import yuyao.models as models

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = mixup_args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch, net, use_cuda, trainloader, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, _) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       mixup_args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(batch_idx, len(trainloader),
                    'Loss: %.3f | Reg: %.5f | TrainAcc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                    100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch, net, use_cuda, testloader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, _) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print(batch_idx, len(testloader),
                    'Loss: %.3f | TestAcc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total,
                    correct, total))

    return (test_loss/batch_idx, 100.*correct.item()/total)



mixup_args = types.SimpleNamespace()
def run_mixup(trainset, testset, num_classes, arch, pretrained, input_channel,n_epoch=200):


    mixup_args.lr = 0.1
    mixup_args.num_gradual = 10
    mixup_args.arch = arch
    mixup_args.epoch = n_epoch
    mixup_args.print_freq = 100
    mixup_args.num_workers = 0
    mixup_args.alpha = 1.
    mixup_args.decay = 1e-4
    mixup_args.batch_size = 128
    mixup_args.num_classes = num_classes
    mixup_args.pretrained = pretrained
    mixup_args.input_channel = input_channel

    use_cuda = torch.cuda.is_available()
    test_acc = 0
    trainloader = DataLoader_noise(trainset,batch_size=mixup_args.batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader_noise(testset, batch_size=100, shuffle=False, num_workers=8)
    print('==> Building model..')
    net = models.__dict__[mixup_args.arch](num_classes=mixup_args.num_classes, input_channel = mixup_args.input_channel, pretrained=mixup_args.pretrained)
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print(torch.cuda.device_count())
        cudnn.benchmark = True
        print('Using CUDA..')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=mixup_args.lr, momentum=0.9,
                        weight_decay=mixup_args.decay)


    for epoch in range(mixup_args.epoch):
        train(epoch, net, use_cuda, trainloader, optimizer, criterion)
        _, curr_test_acc = test(epoch, net, use_cuda, testloader, criterion)

        adjust_learning_rate(optimizer, epoch)

        if mixup_args.epoch-epoch<=10:
            print(epoch)
            test_acc += curr_test_acc

    test_acc = test_acc/10
    return test_acc