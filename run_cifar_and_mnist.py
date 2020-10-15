
import os
import random
import shutil
import time
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import yuyao.models as models
from yuyao.data.data_loader import np_data_loader
from yuyao.utils import AverageMeter, ProgressMeter, fix_seed, accuracy, adjust_learning_rate, save_checkpoint
from yuyao.metrics import l1_error_calculator
from yuyao.data.sampling import subsampling_torch
from yuyao.noise.est_t_matrix import est_t_matrix
from run_trevision import run_trevision
from run_coteaching import run_coteaching
from run_mentoring import run_mentoring
from run_decoupling import run_decoupling
from run_mixup import run_mixup

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', type=str, default='',
                    help='dataset')
parser.add_argument('--arch', type=str, default='',
                    help='model architecture.')
parser.add_argument('--input_channel', type=int, default=1)
parser.add_argument('-j', '--workers', default=0, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('--est_epochs', default=20, type=int,
                    help='number of estimation epochs to run')
parser.add_argument('--cl_epochs', default=200, type=int,
                    help='number of total classification epochs to run')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--trainval_split',  default=0.9, type=float,
                    help='training set ratio')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--lr_decay', default=50, type=int,
                    help='learning rate decay')
parser.add_argument('--flip_rate_fixed', type=float,
                    help='fixed flip rates.')
parser.add_argument('--train_frac', default=1.0, type=float,
                    help='training sample size fraction')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='SGD momentum')
parser.add_argument('--noise_type', type=str, default='symmetric')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training.')
parser.add_argument('--output', type=str, default='./out_dir')
parser.add_argument("--evaluate", action="store_true", 
                    default=False,help="evaluate")
parser.add_argument("--test_acc", action="store_true", 
                    default=False,help="train all methods and display test acc")
args = parser.parse_args()

arch_dict = {"FashionMNIST":"resnet18","cifar10":"resnet18","cifar100":"resnet34","mnist":"Lenet"}
input_channel_dict = {"FashionMNIST":1,"cifar10":3,"cifar100":3,"mnist":1}

def train(train_loader, model, criterion, optimizer, epoch, t_m=np.eye(100)):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.train()
    train_loader.train()
    t_m = torch.Tensor(t_m).cuda()
    for i, (images, n_target, _) in enumerate(train_loader):
        images = images.cuda()
        n_target = n_target.cuda()
        output = model(images)
        probs = F.softmax(output, dim=1)
        probs = torch.matmul(probs,t_m)
        acc1, acc5 = accuracy(probs, n_target, topk=(1, 5))
        output = torch.log(probs+1e-12)
        loss = criterion['cls'](output, n_target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("training acc", top1.avg)


def val(loader, model, criterion, t_m=np.eye(100)):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    loader.eval()
    t_m = torch.Tensor(t_m).cuda()
    with torch.no_grad():
        for i, (images, n_target, _) in enumerate(loader):
            images = images.cuda()
            n_target = n_target.cuda()
            output = model(images)
            probs = F.softmax(output, dim=1)
            probs = torch.matmul(probs,t_m)
            acc1, acc5 = accuracy(probs, n_target, topk=(1, 5))
            output = torch.log(probs+1e-12)
            loss = criterion['cls'](output, n_target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
    return top1.avg


def test(test_loader, model, criterion, out_dir):
    checkpoint = torch.load(os.path.join(out_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    top1 = val(loader = test_loader, model = model, criterion = criterion,t_m = np.eye(args.num_classes))
    return model, checkpoint["best_acc1"], top1


def get_transition_matrices(est_loader, model):
    model.eval()
    est_loader.eval()
    p = []
    T_spadesuit = np.zeros((args.num_classes,args.num_classes))
    with torch.no_grad():
        for i, (images, n_target,_) in enumerate(est_loader):
            images = images.cuda()
            n_target = n_target.cuda()
            pred = model(images)
            probs = F.softmax(pred, dim=1).cpu().data.numpy()
            _, pred = pred.topk(1, 1, True, True)           
            pred = pred.view(-1).cpu().data
            n_target = n_target.view(-1).cpu().data
            for i in range(len(n_target)): 
                T_spadesuit[int(pred[i])][int(n_target[i])]+=1
            p += probs[:].tolist()  
    T_spadesuit = np.array(T_spadesuit)
    sum_matrix = np.tile(T_spadesuit.sum(axis = 1),(args.num_classes,1)).transpose()
    T_spadesuit = T_spadesuit/sum_matrix
    p = np.array(p)
    T_clubsuit = est_t_matrix(p,filter_outlier=True)
    T_spadesuit = np.nan_to_num(T_spadesuit)
    return T_spadesuit, T_clubsuit


def run_forward(train_loader, val_loader, test_loader, args, t_m = np.eye(10), out_dir="./out_dir", epochs=100):
    print(args.arch)
    model = models.__dict__[args.arch](num_classes=args.num_classes,input_channel=args.input_channel, pretrained = False)
    model = model.cuda()
    best_acc1 = 0
    criterion = {'cls': nn.NLLLoss()}
    optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if not args.evaluate:
        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, args.lr, ajust_period=args.lr_decay)
            train(train_loader = train_loader, model = model, criterion = criterion, optimizer = optimizer, epoch = epoch, t_m = t_m)
            acc1 = val(loader = val_loader, model = model, criterion = criterion, t_m = t_m)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint(
                state ={   
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict()
                },
                out=out_dir,
                is_best=is_best
            )
    model, best_val_acc1, test_acc = test(test_loader, model, criterion, out_dir)
    return model, best_val_acc1, test_acc



def load_model(checkpoint_path):
    model = models.__dict__[args.arch](num_classes=args.num_classes,input_channel=args.input_channel, pretrained = False)
    model = model.cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def run_est_T_matrices(est_loader):
    checkpoint_path = os.path.join(args.output+"/"+str(args.seed)+"/est_t", 'model_best.pth.tar')
    print(checkpoint_path)
    model = load_model(checkpoint_path)
    T_spadesuit, T_clubsuit = get_transition_matrices(est_loader, model)
    return T_spadesuit, T_clubsuit


def compose_T_matrices(T_spadesuit, T_clubsuit):
    dual_t_matrix = np.matmul(T_clubsuit, T_spadesuit)
    return dual_t_matrix


def get_noise_rate(t):
    return 1-np.average(t.diagonal())

def main():
  
    if args.seed is not None:
        fix_seed(args.seed)

    train_val_loader, train_loader, val_loader, est_loader, test_loader = np_data_loader(
        dataset = args.dataset,  
        noise_type = args.noise_type,
        random_state = args.seed, 
        batch_size = args.batch_size, 
        add_noise = True, 
        flip_rate_fixed = args.flip_rate_fixed, 
        trainval_split = args.trainval_split,
        train_frac = args.train_frac
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset
    train_val_dataset = train_val_loader.dataset
    true_t_matrix = train_dataset.dataset.t_matrix
    args.num_classes = train_dataset._get_num_classes()
    args.arch = arch_dict[args.dataset]
    args.input_channel = input_channel_dict[args.dataset]

    print(args)
    print(true_t_matrix)
    print("train size",len(train_dataset))
    print("val size",len(val_dataset))
    print("test size",len(test_dataset))
    print("train val size",len(train_val_dataset))
    est_dir = args.output+"/"+str(args.seed)+"/est_t"
   
    print("+++++++start training for estimating transistion matrix T++++++++")
    out_dir = args.output+"/"+str(args.seed)+"/est_t"

    print("++++++++++++run CE+++++++++++++++++")
    # training a network on the noisy sample directly by using cross entropy loss, the transition matrix t_m is set to the identity matrix 
    _, _, ce_acc = run_forward(train_loader =train_loader, val_loader= val_loader, test_loader = test_loader, args = args, t_m = np.eye(args.num_classes), out_dir=out_dir, epochs=args.est_epochs)
    print("CE acc", ce_acc)

    T_spadesuit, T_clubsuit = run_est_T_matrices(est_loader)
    # the emprical noisy class posterior is equal to the intermediate class posterior, therefore T_estimation and T_clubsuit are identical
    T_estimation = T_clubsuit
    T_estimator_err = l1_error_calculator(target = true_t_matrix, target_hat = T_clubsuit)
    print("T-estimator error", T_estimator_err)

    dual_T_estimation = compose_T_matrices(T_spadesuit=T_spadesuit, T_clubsuit = T_clubsuit)
    dual_T_estimator_err = l1_error_calculator(target = true_t_matrix, target_hat = dual_T_estimation)  
    print("DT-estimator error", dual_T_estimator_err)

    if not args.test_acc:
        exit()

    # T_noise_rate = get_noise_rate(T_estimation)
    # DT_noise_rate = get_noise_rate(dual_T_estimation)

    # print("++++++++++++run T-Forward+++++++++++++++++")
    # out_dir = args.output+"/"+str(args.seed)+"/t_foward"
    # _, _, T_forward_acc = run_forward(train_loader = train_loader, val_loader= val_loader, test_loader = test_loader, args = args, t_m = T_estimation, out_dir=out_dir,epochs=args.cl_epochs)
    # print("T-Forward acc", T_forward_acc)

    # print("++++++++++++run DT-Foward+++++++++++++++++")
    # out_dir = args.output+"/"+str(args.seed)+"/dt_foward"
    # _, _, DT_forward_acc = run_forward(train_loader = train_loader, val_loader= val_loader, test_loader = test_loader, args = args, t_m = dual_T_estimation, out_dir=out_dir,epochs=args.cl_epochs)
    # print("DT-Forward acc", DT_forward_acc)  

    # print("++++++++++++run T-Revision+++++++++++++++++")
    # _, _, reweighting_test_acc, _, _, revision_best_test_acc = run_trevision(
    #     train_data = train_dataset, 
    #     val_data = val_dataset, 
    #     test_data = test_dataset, 
    #     input_channel = args.input_channel,
    #     True_T = np.eye(args.num_classes), 
    #     T_hat = T_estimation, 
    #     pretrained = False,
    #     arch=args.arch, 
    #     model_path = est_dir,
    #     n_epoch=args.cl_epochs,
    #     n_epoch_revision=args.cl_epochs,
    #     num_classes = args.num_classes,
    #     dataset = args.dataset,
    #     output = args.output,
    #     seed = args.seed,
    #     evaluate = args.evaluate)
    # print("T-Reweighting acc",reweighting_test_acc)
    # print("T-Revision acc",revision_best_test_acc)


    # print("++++++++++++run DT-Revision+++++++++++++++++")
    # _, _, reweighting_test_acc, _, _, revision_best_test_acc = run_trevision(
    #     train_data = train_dataset, 
    #     val_data = val_dataset, 
    #     test_data = test_dataset, 
    #     input_channel = args.input_channel,
    #     True_T = np.eye(args.num_classes), 
    #     T_hat = dual_T_estimation, 
    #     pretrained = False,
    #     arch=args.arch, 
    #     model_path = est_dir,
    #     n_epoch=args.cl_epochs,
    #     n_epoch_revision=args.cl_epochs,
    #     num_classes = args.num_classes,
    #     dataset = args.dataset,
    #     output = args.output,
    #     seed = args.seed,
    #     evaluate = args.evaluate)
    # print("DT-Reweighting acc",reweighting_test_acc)
    # print("DT-Revision acc",revision_best_test_acc)


    # print("++++++++++++run T-MentorNet+++++++++++++++++")
    # T_mentornet_acc = run_mentoring(train_val_dataset, test_dataset, T_noise_rate, input_channel=args.input_channel, arch=args.arch, pretrained=False, num_classes = args.num_classes, n_epoch = args.cl_epochs)
    # print("T-MentorNet acc", T_mentornet_acc)

    # print("++++++++++++run DT-MentorNet+++++++++++++++++")
    # DT_mentornet_acc = run_mentoring(train_val_dataset, test_dataset, DT_noise_rate, input_channel=args.input_channel, arch=args.arch, pretrained=False, num_classes = args.num_classes, n_epoch = args.cl_epochs)
    # print("DT-MentorNet acc", DT_mentornet_acc)

    # print("++++++++++++run Decoupling+++++++++++++++++") 
    # decoupling_acc = run_decoupling(train_val_dataset, test_dataset, T_noise_rate, input_channel = args.input_channel, arch=args.arch, pretrained=False, num_classes = args.num_classes, n_epoch = args.cl_epochs)
    # print("Decoupling acc", decoupling_acc)

    # print("++++++++++++run Mixup+++++++++++++++++") 
    # mixup_acc = run_mixup(train_val_dataset, test_dataset, num_classes = args.num_classes, input_channel=args.input_channel,  arch=args.arch, pretrained=False, n_epoch = args.cl_epochs)
    # print("Mixup acc", mixup_acc)

    # print("++++++++++++run T-Coteaching+++++++++++++++++") 
    # T_coteaching_acc = run_coteaching(train_val_dataset, test_dataset, T_noise_rate, input_channel = args.input_channel, arch=args.arch, pretrained=False, num_classes = args.num_classes, n_epoch = args.cl_epochs)
    # print("T-Coteaching acc", T_coteaching_acc)   

    # print("++++++++++++run DT-Coteaching+++++++++++++++++")
    # DT_coteaching_acc = run_coteaching(train_val_dataset,test_dataset, DT_noise_rate, input_channel = args.input_channel, arch=args.arch, pretrained=False, num_classes = args.num_classes, n_epoch = args.cl_epochs)
    # print("DT-Coteaching acc", DT_coteaching_acc)



if __name__ == "__main__":
    main()

