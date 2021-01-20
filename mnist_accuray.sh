#!/bin/bash


for train_frac in 1
do
    for seed in 1 2 3 4 5
    do
        for data in mnist
        do
            CUDA_VISIBLE_DEVICES=0 python3 run_cifar_and_mnist.py --dataset ${data} --test_acc --est_epochs 100 --cl_epochs 200 --trainval_split 0.9 --noise_type pairflip --flip_rate_fixed 0.45 --train_frac ${train_frac} --seed ${seed} --output ./output/cl_${data}_tv8_tf${train_frac}_pairflip45
            CUDA_VISIBLE_DEVICES=0 python3 run_cifar_and_mnist.py --dataset ${data} --test_acc --est_epochs 100 --cl_epochs 200 --trainval_split 0.9 --noise_type symmetric --flip_rate_fixed 0.5 --train_frac ${train_frac} --seed ${seed} --output ./output/cl_${data}_tv8_tf${train_frac}_sym50 
            CUDA_VISIBLE_DEVICES=0 python3 run_cifar_and_mnist.py --dataset ${data} --test_acc --est_epochs 100 --cl_epochs 200 --trainval_split 0.9 --noise_type symmetric --flip_rate_fixed 0.2 --train_frac ${train_frac} --seed ${seed} --output ./output/cl_${data}_tv8_tf${train_frac}_sym20
        done
    done
done