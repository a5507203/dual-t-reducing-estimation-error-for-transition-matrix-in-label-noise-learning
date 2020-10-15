import os
import torch
import numpy as np
import argparse, sys
import torch.nn as nn
import torch.optim as optim
from loss import reweight_loss, reweighting_revision_loss
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from yuyao.data.data_loader import np_data_loader
from yuyao.data.data_loader import DataLoader_noise
import types
from yuyao.metrics import l1_error_calculator
import yuyao.models as models




def run_trevision(train_data, val_data, test_data, True_T, 
    T_hat, pretrained, dataset, n_epoch_revision, n_epoch,
    num_classes, arch, model_path, output, seed, evaluate, input_channel):

    revargs = types.SimpleNamespace()
    revargs.lr = 0.01
    revargs.lr_revision = 5e-7
    revargs.weight_decay = 1e-4
    revargs.batch_size = 128
    revargs.n_epoch = n_epoch
    revargs.n_epoch_revision = n_epoch_revision
    revargs.num_classes = num_classes
    revargs.dataset = dataset
    revargs.output = output
    revargs.model_path = model_path
    revargs.seed = seed
    revargs.evaluate = evaluate
    revargs.arch = arch
    revargs.pretrained = pretrained
    revargs.input_channel = input_channel

    if revargs.evaluate:
        revargs.n_epoch = 0
        revargs.n_epoch_revision = 0

    model = models.__dict__[revargs.arch](num_classes=revargs.num_classes, input_channel = revargs.input_channel, pretrained = revargs.pretrained)

    #optimizer and StepLR
    optimizer = optim.SGD(model.parameters(), lr=revargs.lr, weight_decay=revargs.weight_decay, momentum=0.9)
    optimizer_revision = optim.Adam(model.parameters(), lr=revargs.lr_revision, weight_decay=revargs.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
        
    #loss
    loss_func_ce = nn.CrossEntropyLoss()
    loss_func_reweight = reweight_loss()
    loss_func_revision = reweighting_revision_loss()

    #cuda
    if torch.cuda.is_available:
        model = model.cuda()
        loss_func_ce = loss_func_ce.cuda()
        loss_func_reweight = loss_func_reweight.cuda()
        loss_func_revision = loss_func_revision.cuda()
    
    
    checkpoint = torch.load(os.path.join(revargs.model_path, 'model_best_foward.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['epoch'])
    print(os.path.join(revargs.model_path, 'model_best_foward.pth.tar'))
    reweight_out_dir = revargs.output+"/"+str(revargs.seed)+"/reweighting"
    rev_out_dir = revargs.output+"/"+str(revargs.seed)+"/revision"

    if not os.path.isdir(reweight_out_dir):
        os.makedirs(reweight_out_dir)

    if not os.path.isdir(rev_out_dir):
        os.makedirs(rev_out_dir)

    train_loader = DataLoader_noise(dataset=train_data, 
                            batch_size=revargs.batch_size,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False)

    val_loader = DataLoader_noise(dataset=val_data,
                            batch_size=revargs.batch_size,
                            shuffle=False,
                            num_workers=0,
                            drop_last=False)

    test_loader = DataLoader_noise(dataset=test_data,
                            batch_size=revargs.batch_size,
                            num_workers=0,
                            drop_last=False)


    T = torch.from_numpy(T_hat).float().cuda()
    val_acc_list_r = []
    reweighting_val_acc = 0
    reweighting_test_acc = 0
    for epoch in range(revargs.n_epoch):
        
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        eval_loss = 0.
        eval_acc = 0.
        train_loader.train()
        model.train()
        for batch_x, batch_y,_ in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer.zero_grad()
            out = model(batch_x, revision=False)
            prob = F.softmax(out, dim=1)
            prob = prob.t()
            loss = loss_func_reweight(out, T, batch_y)
            out_forward = torch.matmul(T.t(), prob)
            out_forward = out_forward.t()
            train_loss += loss.item()
            pred = torch.max(out_forward, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer.step()
        with torch.no_grad(): 
            model.eval()
            val_loader.eval()
            for batch_x,batch_y,_ in val_loader:
                model.eval()
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                out = model(batch_x, revision=False)
                prob = F.softmax(out, dim=1)
                prob = prob.t()
                loss = loss_func_reweight(out, T, batch_y)
                out_forward = torch.matmul(T.t(), prob)
                out_forward = out_forward.t()
                val_loss += loss.item()
                pred = torch.max(out_forward, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()
        scheduler.step()    

        is_best = val_acc/(len(val_data)) > reweighting_val_acc
        reweighting_val_acc = max(val_acc/(len(val_data)), reweighting_val_acc)
  
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data))*revargs.batch_size, train_acc / (len(train_data))))
        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*revargs.batch_size, val_acc / (len(val_data))))
        val_acc_list_r.append(val_acc / (len(val_data)))
        
        with torch.no_grad():
            model.eval()
            test_loader.eval()
            for batch_x, batch_y,_ in test_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                out = model(batch_x, revision=False)
                loss = loss_func_ce(out, batch_y)
                eval_loss += loss.item()
                pred = torch.max(out, 1)[1]
                eval_correct = (pred == batch_y).sum()
                eval_acc += eval_correct.item()
                
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data))*revargs.batch_size, eval_acc / (len(test_data))))

        if is_best:
            torch.save(model.state_dict(), reweight_out_dir + '/reweight_best.pth')
            reweighting_test_acc = eval_acc/ (len(test_data))

    
    reweight_model_path = reweight_out_dir + '/reweight_best.pth'
    reweight_model_path = torch.load(reweight_model_path)
    model.load_state_dict(reweight_model_path)

    eval_loss = 0.
    eval_acc = 0.
    with torch.no_grad():
        model.eval()
        test_loader.eval()
        for batch_x, batch_y,_ in test_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            out = model(batch_x, revision=False)
            loss = loss_func_ce(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            eval_correct = (pred == batch_y).sum()
            eval_acc += eval_correct.item()
        reweighting_test_acc = eval_acc / (len(test_data))
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data))*revargs.batch_size, reweighting_test_acc))


    nn.init.constant_(model.T_revision.weight, 0.0)
    print()
    print('Revision......')

    revision_estimate_error = 0.
    revision_best_val_acc = 0.
    revision_best_test_acc= 0.

    for epoch in range(revargs.n_epoch_revision):
       
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        eval_loss = 0.
        eval_acc = 0.
        model.train()
        train_loader.train()
        for batch_x, batch_y,_ in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer_revision.zero_grad()  
            out, correction = model(batch_x, revision=True)
            prob = F.softmax(out, dim=1)
            prob = prob.t()
            loss = loss_func_revision(out, T, correction, batch_y)
            out_forward = torch.matmul((T+correction).t(), prob)
            out_forward = out_forward.t()
            train_loss += loss.item()
            pred = torch.max(out_forward, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer_revision.step()

        #val     
        with torch.no_grad(): 
            model.eval()      
            val_loader.eval() 
            for batch_x, batch_y,_ in val_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                out, correction = model(batch_x, revision=True)
                prob = F.softmax(out, dim=1)
                prob = prob.t()
                loss = loss_func_revision(out, T, correction, batch_y)
                out_forward = torch.matmul((T+correction).t(), prob)
                out_forward = out_forward.t()
                val_loss += loss.item()
                pred = torch.max(out_forward, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()
                 
        estimate_error = l1_error_calculator(target = True_T, target_hat = (T+correction).cpu().detach().numpy()) 
        print('Estimate error: {:.6f}'.format(estimate_error))
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data))*revargs.batch_size, train_acc / (len(train_data))))
        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*revargs.batch_size, val_acc / (len(val_data))))

        ## test
        with torch.no_grad():
            model.eval()
            test_loader.eval()
            for batch_x, batch_y,_ in test_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                out, _ = model(batch_x, revision=True)
                loss = loss_func_ce(out, batch_y)
                eval_loss += loss.item()
                pred = torch.max(out, 1)[1]
                eval_correct = (pred == batch_y).sum()
                eval_acc += eval_correct.item()
                
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data))*revargs.batch_size, eval_acc / (len(test_data))))

        is_best = val_acc/(len(val_data)) > revision_best_val_acc
        revision_best_val_acc = max(val_acc/(len(val_data)), revision_best_val_acc)
        if is_best:
            torch.save(model.state_dict(), rev_out_dir + '/revision_best.pth')
            revision_estimate_error = estimate_error
            revision_best_test_acc= eval_acc / (len(test_data))
    
    
    revision_model_path = rev_out_dir + '/revision_best.pth'
    revision_model_path = torch.load(revision_model_path)
    model.load_state_dict(revision_model_path)

    ## test
    eval_loss = 0.
    eval_acc = 0.
    with torch.no_grad():
        model.eval()
        test_loader.eval()
        for batch_x, batch_y,_ in test_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            out, correction = model(batch_x, revision=True)
            loss = loss_func_ce(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            eval_correct = (pred == batch_y).sum()
            eval_acc += eval_correct.item()
            
        revision_best_test_acc = eval_acc / (len(test_data))  

        revision_estimate_error = l1_error_calculator(target = True_T, target_hat = (T+correction).cpu().detach().numpy()) 

    return 0, reweighting_val_acc, reweighting_test_acc, revision_estimate_error, revision_best_val_acc, revision_best_test_acc