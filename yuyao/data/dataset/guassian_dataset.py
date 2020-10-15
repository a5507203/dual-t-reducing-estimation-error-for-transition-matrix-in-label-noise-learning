import numpy as np
import torch.utils.data as Data
import os
from yuyao.noise.generator import CCN_generator
from yuyao.data.data_generator import gaussian_generator_ind
from scipy import stats
from .util import noisify
__all__ =["GAUSSIAN_noise"]

# class GAUSSIAN_noise(Data.Dataset):


#     def __init__(
#         self,
#         transform=None, 
#         target_transform=None, 
#         means=[0,2], 
#         variances=[1,1], 
#         dim=10, 
#         sample_size = 50000,
#         add_noise= True, 
#         symmetric = True, 
#         flip_rate_low = None, 
#         flip_rate_high = None, 
#         flip_rate_fixed = None, 
#         paired = False):
            
#         self.transform = transform
#         self.target_transform = target_transform
#         self.t_matrix = None
#         self.data, self.targets = gaussian_generator_ind(means=means, variances=variances, dim=10, sample_size = sample_size)
#         self.data = stats.zscore(self.data)

        
#         self.clean_targets = self.targets.copy()
#         if add_noise:
#             noisy_targets, t_matrix = CCN_generator(self._get_targets(),low=flip_rate_low, flip_rates = [flip_rate_fixed], high=flip_rate_high,symmetric=symmetric, paired = paired)
#             self.t_matrix = t_matrix
#             self._set_targets(noisy_targets)


#         self.data =  self.data.astype(np.float32)
#         self.clean_targets = np.asarray(self.clean_targets).astype(np.long)
#         self.targets = np.asarray(self.targets).astype(np.long)

#     def __getitem__(self, index):
#         instance, target, clean_target = self.data[index], self.targets[index], self.clean_targets[index]
 
#         if self.transform is not None:
#             instance = self.transform(instance)
            
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#             clean_target = self.target_transform(clean_target)

#         return instance, target, clean_target

        
#     def _set_targets(self,n_targets):
#         print(self.targets)
#         print(n_targets)
#         print(self.t_matrix)
#         self.targets = n_targets


#     def _get_targets(self):
#         return self.targets


#     def __len__(self):
#         return len(self.targets)
        




class GAUSSIAN_noise(Data.Dataset):


    def __init__(
        self,
        transform=None, 
        target_transform=None, 
        means=[0,2], 
        variances=[1,1], 
        dim=10, 
        sample_size = 50000,
        add_noise= True, 
        flip_rate_fixed = None, 
        noise_type = "",
        random_state= 1):
            
        self.transform = transform
        self.target_transform = target_transform
        
        self.data, self.targets, self.clean_posteriors = gaussian_generator_ind(means=means, variances=variances, dim=10, sample_size = sample_size)
        self.data = stats.zscore(self.data)
        self.clean_targets = self.targets.copy()
        nb_classes = self._get_num_classes()
        self.t_matrix = np.eye(nb_classes)
        if add_noise:
            noisy_targets, self.actual_noise_rate, self.t_matrix = noisify(
                dataset=self.data, 
                train_labels=self.targets[:, np.newaxis], 
                noise_type=noise_type, 
                noise_rate=flip_rate_fixed, 
                random_state=random_state,
                nb_classes=nb_classes
            )
            noisy_targets = noisy_targets.squeeze()
            self._set_targets(noisy_targets)
            print("asdf")

        self.data =  self.data.astype(np.float32)
        self.clean_targets = np.asarray(self.clean_targets).astype(np.long)
        self.targets = np.asarray(self.targets).astype(np.long)
        self.noisy_posteriors = np.matmul(self.clean_posteriors, self.t_matrix)

    def __getitem__(self, index):
        instance, target, clean_target, noisy_posterior, clean_posterior = self.data[index], self.targets[index], self.clean_targets[index], self.noisy_posteriors[index], self.clean_posteriors[index]
 
        if self.transform is not None:
            instance = self.transform(instance)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            clean_target = self.target_transform(clean_target)

        return instance, target, clean_target, noisy_posterior, clean_posterior

                
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
        