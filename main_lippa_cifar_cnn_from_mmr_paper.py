import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms
from gurobipy import *
import time
from fgsm_and_lp_utils import my_get_var, my_fgsm
from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent as pgd
import scipy.io as sio
from mlp_b_utils import pgd_restart
from temp_utils import Wong_Conv_mnist_model
import argparse
from temp_utils import test
from cifar_cnn_utils import mat_to_pth,CNN_cifar_from_mat, lp_constructor_cnn, lp_cnn, lippa
import csv


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # trying to force the code to use cpu to see if we can gain time

import sys
model_pth_path = sys.argv[1] 
csv_file_path = sys.argv[2]
EPS_H = 2.0/255.0 #float(sys.argv[3])


############################

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


device = torch.device('cpu') #torch.device("cuda" if use_cuda else "cpu")
print(device)
kwargs = {'pin_memory': True} if use_cuda else {}


train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor()
                    ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

###################################################################################


net = CNN_cifar_from_mat() #Wong_Conv_mnist_model()
net.load_state_dict(torch.load(model_pth_path,map_location=device))



column_name = ["sn_vector" , "eps", "true_label", "y_2nd", "adv_pred","runtime" ]

with open(csv_file_path, mode = 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=';')
    line_count = 0
    csv_writer.writerow(column_name)
    x_adv_dict = {}

    # construction of the root problem
    net_params = net.state_dict()
    net_dict = { key : value.cpu().numpy() for key,value in zip(net_params.keys(),net_params.values()) }
    net_dict['conv1.stride'] = 2 #net.conv1.stride[0]
    net_dict['conv1.padding'] = 1  #net.conv2.padding[0]
    net_dict['conv2.stride'] = 2
    net_dict['conv2.padding']=1
    root_advex_finder = lp_constructor_cnn(net_dict)
    
    for sn in range(1000):
        sample = test_loader.dataset[sn][0].to(device)#.cpu().numpy()
        true_label = torch.tensor(test_loader.dataset[sn][1]).to(device)#.cpu().numpy() # add torch.tensor to prevent an error occuring on another torch version
        net.to(device)
        sample = sample.view(1,3,32,32)
        prob, y_10, z_hat  = net(sample)
        y_torch, y_2nd_torch = torch.argsort(y_10[0],descending=True)[:2]
        y_np = np.int(y_torch.cpu().numpy())
        y_2nd_np = np.int(y_2nd_torch.cpu().numpy())

        #print("iteration ", sn )
        if (sn+1)%100 == 0 :
            print("checkpoint at sample", sn )
            csv_file.flush()
        if (true_label!=y_torch) : # we are not interesting in examples that are already misclassified by the classifier
            continue
        
        pred_dict = {'y_np' : y_np, 'y_2nd_np':y_2nd_np}
        tic = time.time()
        eps, x_adv = lippa(sample, root_advex_finder,lp_cnn, net,pred_dict, eps_fgsm=0.5/256 ,tolerance=0.01, optim_time = 3.0, device = 'cpu',verbose = False, eps_h = EPS_H)
        runtime = time.time()-tic
        
        
        adv_pred = None
        if eps is not None : 
            x_adv = x_adv.view(1,3,32,32)
            _, adv_logit,_ = net(x_adv)
            adv_pred = torch.argmax(adv_logit[0]).cpu().numpy()
            x_adv_dict[sn] = x_adv
        
        line_to_write = [sn, eps, y_np, y_2nd_np,adv_pred, runtime] 
        csv_writer.writerow(line_to_write)
        line_count += 1
        

np.save(csv_file_path,x_adv_dict)     
