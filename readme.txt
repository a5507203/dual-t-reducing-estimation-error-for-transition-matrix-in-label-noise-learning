+++++environment configuration++++++

#########important###############
The code is only tested on Linux based System (Ubuntu 18.04). 
The python version is 3.6.9. The pytorh version is 1.2.0 with GPU acceleration. 

It is unknown that if the code is compatible on windows or different versions of pytorh and python. 
We have not tested to run our code in a CPU environment. 

Upon acception of the paper, we will test the compatibility of the code under different environments and publish the code on GitHub.
To avoid errors caused by inconsistent environment, you are encouraged to run our code under a same environment.

The running time warnings can be ingored. Due to the limited supplementary file size, we only provide mnist dataset. To run the experiments for other datasets, please add the them to the "./datasets" directory. (Note that for CIFAR10 and CIFAR100, it is required to convert the datasets into .npy format) 
#################################



#########run experiments on real world image datasets################
Simple running scripts are provided.


To get Figure 3, open a terminal at the project root directory and type the following commands:

sudo chmod 755 estimation_error.sh
./estimation_error.sh


To get Table 1, open a terminal at the project root directory and type the following commands:

sudo chmod 755 accuracy.sh
./accuracy.sh


To get Table 2, then, open a terminal at the project root directory and type the following commands:

sudo chmod 755 cloth.sh
./cloth1m.sh



