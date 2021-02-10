import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms
import argparse
from gurobipy import *
import matplotlib.pyplot as plt
from temp_utils import *
from fgsm_and_lp_utils import *
import time
from datetime import datetime
import csv
from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent as pgd
from mlp_b_utils import *

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" # trying to force the code to use cpu to see if we can gain time

import sys
model_pth_path = sys.argv[1] 
csv_file_path = sys.argv[2]
EPS_H = float(sys.argv[3])
#test_batch_size = int(sys.argv[4])

# python3 verification_code_of_lippa_mlp_b_utils.py 'model_pth_path' 'csv_file_path' EPS_H test_batch_size

######################## loading the data 

import sys; sys.argv=['']; del sys # added to solve a problem with argparse and ipython

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--threshold', type=float, default=0.01, metavar='N',
                    help='minimum training error to stop epochs')

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device('cpu')  #torch.device("cuda" if use_cuda else "cpu")
print(device)
kwargs = {'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor()
                    ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


net = mlp_b()
#Changing the naming convention to match mine
temp = torch.load(model_pth_path, map_location = device)
keys = net.state_dict().keys()
temp = dict(zip(keys,temp.values()))
net.load_state_dict(temp,)
net.to(device)
#print(net)
####################################

# function computing the number of different linear regions contained in a batch
def linregcount(activations_dict):
    "given the batch activations count the number of linear regions"
    fc1 = activations_dict["fc1"]>=0
    fc2 = activations_dict["fc2"]>=0
    activations =  torch.cat((fc1, fc2), 1)
    row = activations.size()[0]
    number_of_linreg = 1
    for i in range(row-1):
        temp = []
        for j in range(i+1,row):
            diff = sum(activations[i,:]!=activations[j,:])
            temp.append(diff)
            if diff == 0 : 
                break
        if diff == 0:
            continue
        else : 
            number_of_linreg += 1 
    
    return number_of_linreg


# function creating noise that has infinite norm  epsilon  and l2 norm near 90% of the max l2 norm achievable by perturbation of li epsilon
#def my_naive_noise_generator(sample,epsilon,batch_size) : 


def my_noise_generator_0p8(sample,epsilon,batch_size):
    "returns a torch noise vector of shape [batch_size,1,28,28]"
    
    noise = 2*epsilon* (torch.rand((batch_size,28*28))-0.5) # creating the noise to explore the neighborhood of the sample
    
    #l2_norm = torch.norm(noise,p=2,dim=1) # computing the l2 norm of the noise
    
    #noise = noise/l2_norm[:,None]      # making the noise unit vectors, so that the noise can be considered as direction vectors
    
    li_norm = torch.norm(noise,p=float('inf'),dim=1) # computing the l_inf norm of the noise vector
    
    noise = noise/li_norm[:,None]   # making noise have a l_inf norm of 1
    
    noise = epsilon * noise   # finally making the noise have an l_inf norm 

    
    noisy_samples = torch.clamp(sample + noise.view(batch_size,1,28,28), 0,1) # clip in order to make sure all the values are between 0 and 1

    return noisy_samples, noise 
    
def my_signed_noise_generator_max(sample,epsilon,batch_size):

    noise = 2*epsilon* (torch.rand((batch_size,28*28))-0.5)
    noise = epsilon*torch.sign(noise)
    noisy_samples = torch.clamp(sample + noise.view(batch_size,1,28,28), 0,1) # clip in order to make sure all the values are between 0 and 1

    return noisy_samples, noise 


def my_noise_generator(sample,epsilon,batch_size):

    # initial noise li <= epsilon and of l2 around 60% of the maximum achievable by a noise of li = epsilon 
    noise1 = 2*epsilon* (torch.rand((batch_size,28*28))-0.5)
    
    # increase the l2 norm to around 80%
    li_norm = torch.norm(noise1,p=float('inf'),dim=1) # computing the l_inf norm of the noise vector
    noise2 = -epsilon * noise1/li_norm[:,None]   # making noise have a l_inf norm of epsilon and permuting the signs

    # increase the l2 norm to the maximam achievable
    noise3 = epsilon*torch.sign(noise1)
    
    # concatenate all the noises
    noise = torch.cat((noise1,noise2,noise3),0)
    
    noise = noise.to(device)

    noisy_samples = torch.clamp(sample + noise.view(3*batch_size,1,28,28), 0,1) # clip in order to make sure all the values are between 0 and 1

    return noisy_samples, noise


def number_of_instable_relus(activations_dict):

    instable = 0
    fc1 = activations_dict["fc1"]>=0
    fc2 = activations_dict["fc2"]>=0
    activations =  torch.cat((fc1, fc2), 1)
    columns = activations.size()[1]
    
    for i in range(columns):
        unique = torch.unique(activations[:,i])
        #print(unique)
        if unique.size()[0] != 1 : 
            instable += 1



    return instable


vec_num_random_samples =  [64,128,256,512,1024] 
with open(csv_file_path, mode = 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=';')
    column_name = ["sn_vector" , 64,128,256,512,1024, "provably instable ReLUs_2048", "runtime" ]
    csv_writer.writerow(column_name)
    
    
    for sn in range(100) : 
        line_to_write = [sn]
        
        for num_random_samples in vec_num_random_samples:
            tic = time.time()
            sample = test_loader.dataset[sn][0].to(device).reshape(1,1,28,28) # batch_size,channel, height,width
            noisy_samples, noise = my_noise_generator(sample, EPS_H,num_random_samples)

            _,_,activations_dict = net(noisy_samples) # get the activations
            number_of_linreg = linregcount(activations_dict)
            line_to_write.append(number_of_linreg)
            

        noisy_samples, _ = my_noise_generator(sample, EPS_H,2048)
        _,_,activations_dict = net(noisy_samples) # get the activations
        instable = number_of_instable_relus(activations_dict)
        line_to_write.append(instable)
        line_to_write.append(time.time()-tic)
        csv_writer.writerow(line_to_write)


