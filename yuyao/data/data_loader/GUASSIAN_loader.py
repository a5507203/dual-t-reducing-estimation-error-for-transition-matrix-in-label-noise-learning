import torch
import torchvision
import torchvision.transforms as transforms
from .subset import Subset
import yuyao
from yuyao.data.dataset import GAUSSIAN_noise
from yuyao.data.sampling import subsampling_torch
from yuyao.data.data_loader.utils import create_train_val
import numpy as np
from .dataloader import DataLoader_noise

__all__ = ["GUASSIAN_loader"]

def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target 
    
def GUASSIAN_loader(num_workers = 0, sample_size = 50000, batch_size = 128, means=[0,2], variances=[1,1], dim=10, add_noise = False, noise_type = None, flip_rate_fixed = None, random_state = 1, trainval_split=None, train_frac = 1):
    print('=> preparing data..')
    # transform_train = transforms.Compose([transforms.ToTensor()])
    # transform_test = transforms.Compose([transforms.ToTensor()])
    test_dataset = GAUSSIAN_noise(
        target_transform = transform_target,
        add_noise = add_noise, 
        noise_type = noise_type, 
        flip_rate_fixed=flip_rate_fixed,
        sample_size=int(sample_size*(1-trainval_split)),
        random_state = np.random.randint(10000000, size=1)[0],
        means=means, 
        variances=variances, 
        dim=dim
    )
    # test_dataset = GAUSSIAN_noise(add_noise = False,target_transform = transform_target,sample_size=int(sample_size*(1-trainval_split)))
    train_val_dataset =GAUSSIAN_noise(
        target_transform = transform_target,
        add_noise = add_noise,
        noise_type = noise_type, 
        flip_rate_fixed=flip_rate_fixed,
        random_state = random_state,
        means=means, 
        variances=variances, 
        dim=dim, 
        sample_size = sample_size
    )

    train_dataset, val_dataset = create_train_val(train_val_dataset,trainval_split,train_frac)
    train_val_dataset = Subset(train_val_dataset, list(range(0, len(train_val_dataset), 1))) 
    train_val_loader = DataLoader_noise(train_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader_noise(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    est_loader = DataLoader_noise(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader_noise(val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    test_loader = DataLoader_noise(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
  
    return train_val_loader, train_loader, val_loader, est_loader, test_loader
