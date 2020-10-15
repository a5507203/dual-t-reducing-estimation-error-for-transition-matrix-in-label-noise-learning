import torch
from .subset import random_split, Subset
import numpy as np

def create_train_val(dataset, trainval_split, train_frac):
    total_len = len(dataset)
    print(total_len)
    if trainval_split:
        print("split validation set from training data")
        train_size = int(trainval_split * total_len)
        val_size = total_len - train_size
        train_size = int(train_frac*train_size)
        train_dataset, val_dataset,_ = random_split(dataset, [train_size, val_size, total_len-train_size-val_size])  
        
    else:
        print("use training data for validation")
        train_size = int(train_frac*total_len)
        train_dataset, _ = random_split(dataset, [train_size,total_len-train_size])  
        val_dataset = train_dataset
    if type(train_dataset.indices) == torch.Tensor:
        train_dataset.indices = train_dataset.indices.tolist()
    if type(val_dataset.indices) == torch.Tensor:
        val_dataset.indices = val_dataset.indices.tolist()

    return train_dataset, val_dataset



def create_train_val_estY_cloth1m(dataset, trainval_split, train_frac):

    est_dataset_Y = Subset(dataset, dataset.get_clean_targets_idxs())
    print(set(dataset.get_clean_targets_idxs()))
    total_len = len(dataset)
    all_idxs = set(range(0, total_len))
    print(total_len)
    if trainval_split:
        idxs = dataset.get_unknown_targets_idxs()
        print("split validation set from training data")

        train_size = int(trainval_split * total_len)
        val_size = total_len - train_size
        train_size = int(train_frac*train_size)

        val_idxs = np.random.choice(idxs, val_size, replace=False)
        print(val_idxs)
        print(len(val_idxs))
        print(set(dataset.get_clean_targets_by_idxs(est_dataset_Y.indices)))
        exit()
        set(idxs)-set(val_idxs)

        train_idxs = np.array(list(all_idxs - set(val_idxs)))
        train_dataset = Subset(dataset,train_idxs)
        print(train_idxs)
        print(val_idxs)
        exit()
        val_dataset = Subset(dataset,val_idxs)
    else:
        raise NotImplementedError

    if type(train_dataset.indices) ==  torch.Tensor:
        train_dataset.indices = train_dataset.indices.tolist()
    if type(val_dataset.indices) ==  torch.Tensor:
        val_dataset.indices = val_dataset.indices.tolist()
    if type(est_dataset_Y.indices) ==  torch.Tensor:
        est_dataset_Y.indices = val_dataset.indices.tolist()

    return train_dataset, val_dataset, est_dataset_Y