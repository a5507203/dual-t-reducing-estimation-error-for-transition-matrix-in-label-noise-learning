import numpy as np
import torch.utils.data as Data
from PIL import Image
import os
from yuyao.noise.generator import CCN_generator
from .util import noisify
__all__ =["CIFAR10_noise","CIFAR100_noise", "MNIST_noise"]

class CIFAR10_noise(Data.Dataset):


    def __init__(self,
        root="~/.torchvision/datasets/cifar10_npy", 
        train=True, 
        transform=None, 
        transform_eval = None,
        target_transform=None, 
        add_noise= True, 
        flip_rate_fixed = None, 
        noise_type = '', 
        random_state = 1):
            
        self.transform = transform
        self.transform_eval = transform_eval
        self.target_transform = target_transform
        self.t_matrix = None
        root = os.path.expanduser(root)
        self.root = root
        self.apply_transform_eval = False
        if train:
            self.images = np.load(os.path.join(root, 'train_images.npy'))
            self.targets = np.load(os.path.join(root, 'train_labels.npy'))
            self.clean_targets = np.load(os.path.join(root, 'train_labels.npy'))
        else:
            self.images = np.load(os.path.join(root, 'test_images.npy'))
            self.targets = np.load(os.path.join(root, 'test_labels.npy'))
            self.clean_targets = np.load(os.path.join(root, 'test_labels.npy'))
        self.images = self.images.reshape((-1,3,32,32))
        self.images = self.images.transpose((0, 2, 3, 1)) 

        if add_noise:
            noisy_targets, self.actual_noise_rate, self.t_matrix = noisify(
                dataset=self.images, 
                train_labels=self.targets[:, np.newaxis], 
                noise_type=noise_type, 
                noise_rate=flip_rate_fixed, 
                random_state=random_state,
                nb_classes=self._get_num_classes()
            )
            noisy_targets = noisy_targets.squeeze()
            self._set_targets(noisy_targets)


    def __getitem__(self, index):
        img, label, clean_label = self.images[index], self.targets[index], self.clean_targets[index]
        img = Image.fromarray(img)

        if self.apply_transform_eval:
            transform = self.transform_eval
        else:
            transform = self.transform  

        if self.transform is not None:
            img = transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            clean_label = self.target_transform(clean_label)

        return img, label, clean_label
     
    def _set_targets(self,n_targets):
        self.targets = n_targets

    def _get_num_classes(self):
        return len(set(self.targets))

    def _get_targets(self):
        return self.targets

    def eval(self):
        self.apply_transform_eval = True

    def train(self):
        self.apply_transform_eval = False

    def __len__(self):
        return len(self.targets)

class CIFAR100_noise(Data.Dataset):


    def __init__(self, 
        root="~/.torchvision/datasets/cifar100_npy", 
        train=True, 
        transform = None, 
        transform_eval = None,
        target_transform = None, 
        add_noise= True, 
        flip_rate_fixed = None, 
        noise_type = 'symmetric', 
        random_state = 1):

        self.transform = transform
        self.transform_eval = transform_eval
        self.target_transform = target_transform
        self.t_matrix = None
        root = os.path.expanduser(root)
        self.root = root
        self.apply_transform_eval = False
        if train:
            self.images = np.load(os.path.join(root, 'train_images.npy'))
            self.targets = np.load(os.path.join(root, 'train_labels.npy'))
            self.clean_targets = np.load(os.path.join(root, 'train_labels.npy'))
        else:
            self.images = np.load(os.path.join(root, 'test_images.npy'))
            self.targets = np.load(os.path.join(root, 'test_labels.npy'))
            self.clean_targets = np.load(os.path.join(root, 'test_labels.npy'))
        self.images = self.images.reshape((-1,3,32,32))
        self.images = self.images.transpose((0, 2, 3, 1)) 

        if add_noise:
            noisy_targets, self.actual_noise_rate, self.t_matrix = noisify(
                dataset=self.images, 
                train_labels=self.targets[:, np.newaxis], 
                noise_type=noise_type, 
                noise_rate=flip_rate_fixed, 
                random_state=random_state,
                nb_classes=self._get_num_classes()
            )
            noisy_targets = noisy_targets.squeeze()
            self._set_targets(noisy_targets)



    def __getitem__(self, index):
        img, label, clean_label = self.images[index], self.targets[index], self.clean_targets[index]
        img = Image.fromarray(img)
        if self.apply_transform_eval:
            transform = self.transform_eval
        else:
            transform = self.transform     

        if transform is not None:
            img = transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            clean_label = self.target_transform(clean_label)

        return img, label, clean_label

        
    def _set_targets(self,n_targets):
        self.targets = n_targets

    def _get_num_classes(self):
        return len(set(self.targets))

    def _get_targets(self):
        return self.targets

    def eval(self):
        self.apply_transform_eval = True

    def train(self):
        self.apply_transform_eval = False

    def __len__(self):
        return len(self.targets)



class MNIST_noise(Data.Dataset):

    def __init__(self, 
        root="~/.torchvision/datasets/FashionMINIST_npy", 
        train=True, 
        transform=None, 
        transform_eval = None,
        target_transform=None, 
        add_noise= True, 
        flip_rate_fixed = None, 
        noise_type = 'symmetric', 
        random_state = 1):

        self.transform = transform
        self.transform_eval = transform_eval
        self.target_transform = target_transform
        self.t_matrix = None
        root = os.path.expanduser(root)
        self.root = root
        self.apply_transform_eval = False
        if train:
            self.images = np.load(os.path.join(self.root, 'train_images.npy'))
            self.targets = np.load(os.path.join(self.root, 'train_labels.npy'))
            self.clean_targets = np.load(os.path.join(self.root, 'train_labels.npy'))
        else:
            self.images = np.load(os.path.join(self.root, 'test_images.npy'))
            self.targets = np.load(os.path.join(self.root, 'test_labels.npy')) - 1
            self.clean_targets = np.load(os.path.join(self.root, 'test_labels.npy')) - 1
            

        if add_noise:
            noisy_targets, self.actual_noise_rate, self.t_matrix = noisify(
                dataset=self.images, 
                train_labels=self.targets[:, np.newaxis], 
                noise_type=noise_type, 
                noise_rate=flip_rate_fixed, 
                random_state=random_state,
                nb_classes=self._get_num_classes()
            )
            noisy_targets = noisy_targets.squeeze()
            self._set_targets(noisy_targets)


    def __getitem__(self, index):
        img, label, clean_label = self.images[index], self.targets[index], self.clean_targets[index]
        img = Image.fromarray(img)

        if self.apply_transform_eval:
            transform = self.transform_eval
        else:
            transform = self.transform     

        if transform is not None:
            img = transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            clean_label = self.target_transform(clean_label)

        return img, label, clean_label

        
    def _set_targets(self,n_targets):
        self.targets = n_targets

    def _get_num_classes(self):
        return len(set(self.targets))

    def _get_targets(self):
        return self.targets

    def eval(self):
        self.apply_transform_eval = True

    def train(self):
        self.apply_transform_eval = False

    def __len__(self):
        return len(self.targets)

        