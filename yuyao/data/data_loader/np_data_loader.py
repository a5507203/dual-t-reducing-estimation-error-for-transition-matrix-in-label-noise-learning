import torch
import torchvision
import torchvision.transforms as transforms
from .subset import Subset
import yuyao
from yuyao.data.data_loader.utils import create_train_val
from .dataloader import DataLoader_noise
import numpy as np

__all__ = ["np_data_loader"]

data_info_dict = {
    "CIFAR10":{
        "mean":(0.49139968,0.48215841,0.44653091),
        "std":(0.24703223,0.24348513,0.26158784),
        "root": "~/.torchvision/datasets/cifar10_npy",
        'random_crop':32
    },
    "CIFAR100":{
        # "mean":(0.50707516,0.48654887,0.44091784),
        # "std":(0.26733429,0.25643846,0.27615047),
        "mean":(0.4914, 0.4822, 0.4465),
        "std":(0.2023, 0.1994, 0.2010),
        "root": "~/.torchvision/datasets/cifar100_npy",
        'random_crop':32
    },
    "MNIST":{
        "mean":(0.1306604762738429,),
        "std":(0.30810780717887876,),
        "root": "~/.torchvision/datasets/mnist_npy",
        'random_crop':None
    },
    "FASHIONMNIST":{
        "mean":(0.286,),
        "std":(0.353,),
        "root": "~/.torchvision/datasets/FashionMNIST",
        'random_crop':28
    }
}

def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target  

def np_data_loader(dataset = "CIFAR10", num_workers = 0, batch_size = 128, add_noise = False, noise_type = None, flip_rate_fixed = None, random_state = 1, trainval_split=None, train_frac = 1):
    print('=> preparing data..')
    t_matrix = []
    dataset = dataset.upper()
    info = data_info_dict[dataset]

    root = info["root"]
    random_crop = info["random_crop"]
    normalize = transforms.Normalize(info["mean"], info["std"])
    if(random_crop != None):
        transform_train = transforms.Compose([transforms.RandomCrop(random_crop, padding=4),transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    else:
        transform_train = transforms.Compose([transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    test_dataset = yuyao.data.dataset.__dict__[dataset+"_noise"](root=root, train=False, transform=transform_test, transform_eval=transform_test, add_noise = False, target_transform = transform_target)
    train_val_dataset = yuyao.data.dataset.__dict__[dataset+"_noise"](
        root = root, 
        train = True, 
        transform = transform_train, 
        transform_eval = transform_test,
        target_transform = transform_target,
        add_noise = True,
        noise_type = noise_type, 
        flip_rate_fixed = flip_rate_fixed,
        random_state = random_state
    )

    train_dataset, val_dataset = create_train_val(train_val_dataset,trainval_split,train_frac)
    train_val_dataset = Subset(train_val_dataset, list(range(0, len(train_val_dataset), 1))) 
    train_val_loader = DataLoader_noise(train_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader_noise(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader_noise(val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    est_loader = DataLoader_noise(train_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    test_loader = DataLoader_noise(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    
    return train_val_loader, train_loader, val_loader, est_loader, test_loader



