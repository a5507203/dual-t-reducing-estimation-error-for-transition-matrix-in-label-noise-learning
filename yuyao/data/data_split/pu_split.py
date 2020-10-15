import pandas as pd 
import numpy as np
import random

__all__ = ["random_pu_split"]

def random_pu_split(data, positive_label, positive_frac = 0.25):
    r"""
    split a data array to positive and unlabel data w.r.t. the fraction of positive dataset.
    Note that, this function will not change the examples' labels 

    Arguments:
        data (array): the whole dataset, with the label at the last column
        positive_label (int): the label choose to be the postive

    Returns:
        unlabeled data (array): the labels are not changed
        positive data (array)
    """
    df = pd.DataFrame(data)
    label_column_idx= len(df.columns)-1 
    
    pos_labels_idxs = list(df.loc[df.loc[:,label_column_idx]==positive_label,label_column_idx].index)
    number_of_pos_examples = int(len(pos_labels_idxs)*positive_frac)
    pos_idxs = random.sample(pos_labels_idxs,number_of_pos_examples)
    mixture_idxs = list(set(df.index)-set(pos_idxs))

    return df.iloc[mixture_idxs,:].values.tolist(), df.iloc[pos_idxs,:].values.tolist()



# def to_binary_classification(df):
#     label_index = df.columns[-1]
#     df.loc[df.loc[:,label_index]>1,label_index]=1
#     return df

# def relabel(data, new_label = 0):
#     arr = np.array(data)
#     arr[:,-1]=new_label
#     return arr.tolist()

# def norm_data(arr):
#     a = np.array(arr)
#     print(np.mean(a, axis=0))
#     return ((a - np.mean(a,axis=0)) / np.std(a,axis=0)).tolist()




# def random_split_class_balanced_data(sample, train_set_frac = 0.5, val_set_frac = 0.5):
#     dataset = DatasetArray(data=sample)
#     train_idxs = []
#     val_idxs = []

#     unique_labels, n_samples_per_class = np.unique(dataset.label_arr,return_counts=True)
#     inner_class_idxs_list = []

#     for i in range(len(unique_labels)):
        
#         count = 0
#         labels = dataset.label_arr
#         label = unique_labels[i]
#         n = n_samples_per_class[i]

#         n_train_and_val = int(n*(train_set_frac+val_set_frac))

#         n_train = int(n*(train_set_frac))    
#         #randomly choose some indices 
#         inner_class_idxs_list = np.random.choice(n, n_train_and_val, replace = False)
#         inner_class_idxs_train = inner_class_idxs_list[:n_train]
#         inner_class_idxs_val= inner_class_idxs_list[n_train:]

#         for i in range(len(labels)):
#             curr_label = labels[i]
#             if curr_label == label:
#                 if(count in inner_class_idxs_train):
#                     train_idxs.append(i)
#                 elif(count in inner_class_idxs_val):
#                     val_idxs.append(i)
#                 count += 1
        
#     return Subset(dataset, train_idxs),Subset(dataset, val_idxs)