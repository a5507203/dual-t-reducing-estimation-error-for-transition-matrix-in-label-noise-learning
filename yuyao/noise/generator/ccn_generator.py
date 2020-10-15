
import random
import numpy as np
from yuyao.data.sampling import subsampling
from yuyao.data.data_split import random_split_array

__all__ = ["CCN_generator"]

def CCN_generator(clean_labels=None, flip_rates=None, transition_matrix=None, low=None, high=None, symmetric=False, paired=False):
    print(clean_labels)
    if type(transition_matrix) is np.ndarray:
        return CCN_generator_t_matrix(clean_labels, transition_matrix)

    elif low != None and high != None:
        return CCN_generator_range(clean_labels, low, high, symmetric, paired)

    elif type(flip_rates) is np.ndarray or flip_rates != None:
        return CCN_generator_flip_rates(clean_labels, flip_rates, symmetric, paired)



def binary_CCN_generator(clean_labels, flip_rates=[0.15,0.25]):
    transition_matrix  = [[1-flip_rates[0],flip_rates[0]],[flip_rates[1],1-flip_rates[1]]]
    noisy_labels = CCN_generator(clean_labels, transition_matrix)
    return noisy_labels
   


def CCN_generator_range(clean_labels, low=0.5, high=0.9, symmetric=False, paired=False):
    num_labels = len(set(clean_labels))
    flip_rates=np.random.uniform(low=low, high=high, size=num_labels)
    return CCN_generator_flip_rates(clean_labels, flip_rates=flip_rates, symmetric=symmetric, paired=paired)



def CCN_generator_flip_rates_paired(clean_labels, flip_rates=[0.6]):
    num_labels = len(set(clean_labels))
    pesudo_labels = [i for i in range(num_labels)]
    transition_matrix = []
    if len(flip_rates)==1:
        flip_rates = flip_rates*num_labels  
    for source_label in pesudo_labels:
        curr_flip_rate = flip_rates[source_label]
        target_label = np.random.choice(list(set(pesudo_labels)-set([source_label])), 1,replace=False)
        random_flip_rates = np.zeros(num_labels)
        random_flip_rates[source_label] = 1-curr_flip_rate
        random_flip_rates[target_label] = curr_flip_rate
        transition_matrix.append(random_flip_rates)  
    transition_matrix = np.array(transition_matrix)
    print(transition_matrix)
    return CCN_generator_t_matrix(clean_labels, transition_matrix), np.array(transition_matrix)



def CCN_generator_flip_rates(clean_labels, flip_rates=[0.6], symmetric=False, paired=False):
    if paired:
        return CCN_generator_flip_rates_paired(clean_labels, flip_rates=flip_rates)

    num_labels = len(set(clean_labels))
    
    transition_matrix = []
    if len(flip_rates)==1:
        flip_rates = flip_rates*num_labels
    # print(flip_rates)
    print(num_labels)
    for i in range(num_labels):
        flip_rate = flip_rates[i]
        if not symmetric:
            random_flip_rates = np.random.randint(0,10000,num_labels-1)
            # print(random_flip_rates)
            random_flip_rates = list(random_flip_rates/sum(random_flip_rates)*flip_rate)
        else:
            random_flip_rates = [flip_rate/(num_labels-1)]*(num_labels-1)
        random_flip_rates.insert(i,1-sum(random_flip_rates))
        transition_matrix.append(random_flip_rates)

    return CCN_generator_t_matrix(clean_labels, transition_matrix), np.array(transition_matrix)


def check_no_extreme_noise(t):
    for i in range(len(t)):
        if np.argmax(t[i]) != i:
            return False 
    return True

def CCN_generator_random(clean_labels, flip_rates=[0.9]):

    num_labels = len(set(clean_labels))
    transition_matrix = []
    if len(flip_rates)==1:
        flip_rates = flip_rates*num_labels
    # print(flip_rates)
    print(num_labels)
    while True:
        for i in range(num_labels):
            flip_rate = flip_rates[i]
            random_flip_rates = np.random.randint(0,10000,num_labels-1)
            random_flip_rates = list(random_flip_rates/sum(random_flip_rates)*flip_rate)
            random_flip_rates.insert(i,1-sum(random_flip_rates))
            transition_matrix.append(random_flip_rates)
        if check_no_extreme_noise(transition_matrix):
            break

    return CCN_generator_t_matrix(clean_labels, transition_matrix), np.array(transition_matrix)



def CCN_generator_t_matrix(clean_labels, transition_matrix):

    unique_labels = list(set(clean_labels))
    assert(len(transition_matrix) == len(transition_matrix[0]))
    assert(len(unique_labels) == len(transition_matrix))
    clean_labels = np.array(clean_labels)
    noisy_labels = np.array(clean_labels)

    for i in range(len(unique_labels)):

        curr_label = int(unique_labels[i])
        curr_label_flip_rates = transition_matrix[curr_label]
        curr_label_idxs = np.where(clean_labels==curr_label)[0]
        split_curr_label_idxs = random_split_array(arr=curr_label_idxs,split_ratios=curr_label_flip_rates)

        for j in range(len(unique_labels)):
            noisy_label = unique_labels[j]
            if i != j:
                noisy_labels[split_curr_label_idxs[j]] = noisy_label
                
    return noisy_labels


        
