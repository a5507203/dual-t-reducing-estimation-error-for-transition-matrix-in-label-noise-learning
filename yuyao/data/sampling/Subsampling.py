import numpy as np
import random


__all__ = ["subsampling", "subsampling_torch"]

def subsampling(samples, class_piror=[0.5, 0.5], total_sample_length=400):
    subsamples = []
    unusedsamples = []
    for i in range(len(samples)):
        sample = samples[i]
        n = len(sample)
        idxs = [j for j in range(n)]
        subsample_idxs = np.random.choice(n, int(total_sample_length*class_piror[i]), replace = True)
        unusedsamples_idxs = list(set(idxs)-set(subsample_idxs))

        subsample = [sample[idx] for idx in subsample_idxs]
        unusedsample = [sample[idx] for idx in unusedsamples_idxs]

        subsamples.append(subsample)
        unusedsamples.append(unusedsample)
   
    return subsamples, unusedsamples



# subsampling maintaining class piror returns indices. It can be used for torch Subset object 

def subsampling_torch(targets, ratio=0.8):
   
    unused_idx = []
    sampled_idxs = []
   
    values, counts = np.unique(ar = targets,  return_counts = True)

    for val in values:
        idxs = np.where(targets == val)[0]

        subsample_idxs = np.random.choice(idxs,int(len(idxs)*ratio), replace = False)
 
        sampled_idxs += list(subsample_idxs)
        unused_idx += list(set(idxs) - set(subsample_idxs))

    return sampled_idxs, unused_idx

