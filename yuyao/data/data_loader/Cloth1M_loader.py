import torch
import torchvision
import torchvision.transforms as transforms
from .subset import Subset
from yuyao.data.dataset import ImageFolder_noise
from .utils import create_train_val
from .dataloader import DataLoader_noise

__all__ = ["Cloth1M_loader"]


def Cloth1M_loader(num_workers = 0, batch_size = 128, trainval_split=None, train_frac=1):
    print('=> preparing data..')
    
    transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])
    transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])  

    test_dataset = ImageFolder_noise(root="./datasets/clothing1m/clean_test", transform=transform_test, transform_eval=transform_test, avoid_io=False)
    train_val_dataset = ImageFolder_noise(root="./datasets/clothing1m/noisy_train", transform=transform_train, transform_eval=transform_test, avoid_io=False)
  
  
    train_dataset, val_dataset = create_train_val(train_val_dataset,trainval_split,train_frac)
    train_val_dataset = Subset(train_val_dataset, list(range(0, len(train_val_dataset), 1)))  
    train_val_loader = DataLoader_noise(train_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader_noise(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader_noise(val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    est_loader = DataLoader_noise(train_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    test_loader = DataLoader_noise(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)

    return train_val_loader, train_loader, val_loader, est_loader, test_loader