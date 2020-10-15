import numpy as np
import random



__all__ = ["random_split_array"]
def random_split_array(arr, split_ratios):

    arr_n = len(arr)
    start_idx = 0
    np.random.shuffle(arr)
    split_arr = []
    for r in split_ratios:
        end_idx = start_idx + int(arr_n*r)
        split_arr.append(arr[start_idx:end_idx])
        start_idx = end_idx
    return split_arr
