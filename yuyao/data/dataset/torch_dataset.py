import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

__all__ = ['DatasetCSV', 'DatasetArray', 'DatasetArray_noise']

class DatasetCSV(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data_info = pd.read_csv(file_path,header=None)
        label_index = self.data_info.columns[-1]
        self.data_arr = np.asarray(self.data_info.iloc[:, self.data_info.columns != label_index]).astype(np.float32)
        self.label_arr = np.asarray(self.data_info.iloc[:, label_index])
        self.transform = transform

        
    def __len__(self):
        return len(self.data_arr)
    
    def __getitem__(self, index):
     
        data = self.data_arr[index]
        label = self.label_arr[index]
        
        if self.transform is not None:
            data = self.transform(data)
            
        return (data, label)



class DatasetImgArray(Dataset):
    
    def __init__(self, data, labels, transform=None):
      
        self.data_arr = np.asarray(data)
        self.label_arr = np.asarray(labels).astype(np.long)

        self.transform = transform

        
    def __len__(self):
        return len(self.data_arr)
    
    def __getitem__(self, index):
     
        data = self.data_arr[index]
        label = self.label_arr[index]
        
        if self.transform is not None:
            data = self.transform(data)
            
        return (data, label)



# class DatasetArray(Dataset):
    
#     def __init__(self, data, labels=None, transform=None):
#         if isinstance(labels, np.ndarray) or labels:
#             self.data_arr = np.asarray(data).astype(np.float32)
#             self.label_arr = np.asarray(labels).astype(np.long)
#         else:
#             tmp_arr = np.asarray(data)
#             self.data_arr = tmp_arr[:,:-1].astype(np.float32)
#             self.label_arr = tmp_arr[:,-1].astype(np.long)
#         self.transform = transform

        
#     def __len__(self):
#         return len(self.data_arr)
    
#     def __getitem__(self, index):
     
#         data = self.data_arr[index]
#         label = self.label_arr[index]
        
#         if self.transform is not None:
#             data = self.transform(data)
            
#         return (data, label)
        

class DatasetArray(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data =  np.asarray(data).astype(np.float32)
        self.targets = np.asarray(targets).astype(np.long)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.targets)


class DatasetArray_noise(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data =  np.asarray(data).astype(np.float32)
        self.transform = transform
        self.clean_targets = np.asarray(targets).astype(np.long)
        self.targets = np.asarray(targets).astype(np.long)
    def __getitem__(self, index):
        x = self.data[index]
        n_targets = self.targets[index]
        c_targets = self.clean_targets[index]
        if self.transform:
            x = self.transform(x)
        
        return x, n_targets, c_targets
    
    def __len__(self):
        return len(self.targets)
