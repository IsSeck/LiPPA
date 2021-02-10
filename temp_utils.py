
import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms
import argparse
from gurobipy import *

# just to train a MLP and save the value of the weights so that it can be reused later

d = 784 # number of pixels on my image
e = 256 # number of neurons on the hidden layers
c = 10 # number of classes


class Net(nn.Module): # expected input size 28x28

    def __init__(self, prob_only=False):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(d,e) # an affine operation :y = Wx + b
        self.fc2 = nn.Linear(e,c)
        self.prob_only = prob_only

    def forward(self, x):
        bs = x.size()[0] # batch_size
        x = x.view(bs,-1) # flattening the output for the fully connected layer.
        fc1 = self.fc1(x)
        fc1_relu = F.relu(fc1)
        output = self.fc2(fc1_relu)
        #output = torch.squeeze(output)
        if self.prob_only :
            return F.softmax(output, dim = -1)
        else :
            return F.softmax(output, dim = -1), output, fc1 # return prob and logit, the "probability" of belonging to one class and the logit used to compute those probabilities


    def num_flat_features(self, x):
        size = x.size() # all dimension of x which is a parameter of the network
        num_features = 1

        for s in size :
            num_features *= s
        return num_features

    def total_num_parameters(self):
        params = list(self.parameters())
        num_params = 0

        for p in params:
            #print(self.num_flat_features(p))
            num_params += self.num_flat_features(p)

        return num_params

def train(args, model, device, train_loader, optimizer):

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad_(True)
        optimizer.zero_grad()

        prob , logit, _ = model(data)
        loss = F.cross_entropy(logit, target) # verified see torch.nnCrossEntropyLoss
        loss.backward()
        optimizer.step()

    return loss.item()



def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _ , logit, _ = model(data)
            test_loss += F.cross_entropy(logit, target, reduction='sum').item() # sum up batch loss
            pred = logit.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return 100. * correct / len(test_loader.dataset)


class ConvNet(nn.Module): # expected input size 28x28

    def __init__(self, prob_only=False):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(1,8,5) # an affine operation :y = Wx + b
        self.fc1 = nn.Conv2d(8,10,24)
        self.prob_only = prob_only
    def forward(self, x):
        bs = x.size()[0] # batch_size
        conv1 = self.conv1(x)
        conv1_relu = F.relu(conv1)
        output = self.fc1(conv1_relu)
        output = output.view(bs,-1)
        output = torch.squeeze(output)
        if prob_only:
            return F.softmax(output, dim = -1)
        else : 
            return F.softmax(output, dim = -1), output,conv1 # return prob and logit, the "probability" of belonging to one class and the logit used to compute those probabilities, and the result of the convolution


    def num_flat_features(self, x):
        size = x.size() # all dimension of x which is a parameter of the network
        num_features = 1

        for s in size :
            num_features *= s
        return num_features

    def total_num_parameters(self):
        params = list(self.parameters())
        num_params = 0

        for p in params:
            #print(self.num_flat_features(p))
            num_params += self.num_flat_features(p)

        return num_params



def interval_approximation_bound(w,b, lb,ub):
    """
    using arithmetic, compute the upper and lower bound of wx+b where   lb <= x <= ub,
    params : 
        w : weights, a vector of weights;
        b : bias, the bias term;
        lb : lower bound of x;
        ub : upper bound of x; 
 
    
    returns: 
        upperbound : upper bound of wx+b, where lb <= x <= ub, computed with interval arithmetic
        lowerbound : lower bound of wx+b, where lb <= x <= ub, computed with interval arithmetic
    """

    w_plus = np.maximum(0, w)
    w_minus = np.minimum(0, w)

    upperbound = w_plus @ ub + w_minus @ lb + b
    lowerbound = w_minus @ ub + w_plus @ lb + b

    return lowerbound, upperbound


def interval_approximation_bound_conv(filters,bias,lb,ub, patch_size) :
    """
    using arithmetic, compute the upper and lower bound of conv(x,filters) where   lb <= x <= ub,
    params : 
        filters : the ndarray of filters;
        bias : bias, the bias term;
        lb : lower bound of x, in CHW format;
        ub : upper bound of x, in CHW format; 
 
    
    returns: 
        upperbound : upper bound of conv(x,filters), where lb <= x <= ub, computed with interval arithmetic
        lowerbound : lower bound of conv(x,filters), where lb <= x <= ub, computed with interval arithmetic
    """
    nf, nc, hf, wf  = filters.shape
    _, h, w = lb.shape
    output_shape = [nf,h-(hf//2)*2, w-(wf//2)*2]
    filters_2d = filters.reshape(nf,-1)
    
    
    
    f_plus = np.maximum(0, filters_2d)
    f_minus = np.minimum(0,filters_2d)
    
    lb_patch = my_patch_extractor(lb,hw=patch_size).T
    ub_patch = my_patch_extractor(ub,hw=patch_size).T
    
    #print( "f_plus.shape=", f_plus.shape)
    
    upperbound = f_plus @ ub_patch + f_minus @ lb_patch  
    lowerbound = f_minus @ ub_patch + f_plus @ lb_patch
    
    for i in range(nf):
        upperbound[i,:] += bias[i]
        lowerbound[i,:] += bias[i]
        
    #print("output_shape :", output_shape)
    return lowerbound.reshape(output_shape), upperbound.reshape(output_shape)

def my_list(I):
    return [tuple(i) for i in I]

def my_coord(n1,n2,n3):
    return [(i,j,k) for i in range(n1) for j in range(n2) for k in range(n3)]

def bound_dict(I,M):
    """
    input : 
            + I, a list of tuples
            + M, the matrice containing the bound
    output : 
            + bd, a dictionary with keys I and values M(I) 
    """
    bd = {}
    for i in I:
        bd[i] = M[i]
    
    return bd

def gurobi_conv(img_grb,chw,filters, optim_model, conv):
    """
    inputs :
        optim_model : the optimization model we use to create the
        img_grb     : an input "image" in the format CHW, a gurobi variable
        chw         : the shape of img_grb
        filters     : the filters in format CHW, a numpy variable, odd number of pixel
        conv        : the variable which will contain the linear expression of the convolution

    output :
        conv : the convolutions written as linear expression that can be used by gurobi

    """

    c,h,w = chw
    nf,cf, hf, wf = filters.shape
    t = hf//2 # we suppose that hf=wf, then we can use the same value to compute where the convolutions starts and where it ends

    #print("t",t)
    #print("h-t",h-t)
    
    #print ("cf", cf, "\t c", c)
    if cf != c :
        print("the filters and the images do not have the same dimension ")
        return -1
    
    for k in range(nf): # on the filters
        for l in range(c) : # on the channels
            for i in range(t,h-t): # on the height
                for j in range(t,w-t): # on the width
                    conv[k,i-t,j-t] =  quicksum( [ img_grb[m,i-t+n,j-t+u]*filters[k,m,n,u] for m in range(c) for n in range(hf)  for u in range(wf)  ] )
     
                    #print("indices",k, i-t,j-t)
    #print("img_grb", img_grb)
    #print("filters", filters)
    return conv

def gurobi_conv_egal(img_grb,chw,filters, optim_model, conv):
    """
    filters and image have the same height and width.
    inputs :
        optim_model : the optimization model we use to create the
        img_grb     : an input "image" in the format CHW, a gurobi variable
        chw         : the shape of img_grb
        filters     : the filters in format CHW, a numpy variable
        conv        : the variable which will contain the linear expression of the convolution
        
    output :
        conv : the convolutions written as linear expression that can be used by gurobi

    """

    c,h,w = chw
    nf,cf, hf, wf = filters.shape
    t = hf//2 # we suppose that hf=wf, then we can use the same value to compute where the convolutions starts and where it ends

    #print("t",t)
    #print("h-t",h-t)
    
    #print ("cf", cf, "\t c", c)
    if cf != c :
        print("the filters and the images do not have the same dimension ")
        return -1
    
    for k in range(nf): # on the filters
        for l in range(c) : # on the channels
            conv[k] =  quicksum( [ img_grb[m,n,u]*filters[k,m,n,u] for m in range(c) for n in range(hf)  for u in range(wf)  ] )

    return conv

def my_patch_extractor(img,hw=5):
    """
        inputs : 
            img : an image in the format CHW
            hw  : the size of the square patch, and hw is an odd number
        
        output :
            patch_matrix : a matrix containing all the patch, its dimension is (-1, C*hw*hw), and it is arranged channel by channel on each row
                           and the rows (h-hw//2) first rows represents the patch on the first row. Patches are extracted row-wise  

    """

    # getting the dimensions of the image
    #print(img.shape)
    c,h,w = img.shape
    t = (hw-1)//2
    n_patch = (h-2*t)*(w-2*t)
    patch_matrix = np.zeros((n_patch,c*hw*hw))

    k = 0
    for i in range(t,h-t):
        for j in range(t,w-t):
            k = (i-t)*(w-2*t)+j-t
            patch_matrix[k] = img[:,i-t:i+t+1,j-t:j+t+1].reshape(-1)
            

    #return c,h,w
    return patch_matrix 

def grb_padding(img_grb, chw, padding=1 ) : 
    
    c,h,w = chw
    
    pad_coord = my_coord(c,h+2*padding,w+2*padding)
    
    padded = dict.fromkeys(pad_coord, 0.0)
    
    
    for k in range(c): # on the channels
        for i in range(h): # on the height
                for j in range(w): # on the width
                    padded[k,i+padding,j+padding] = img_grb[k,i,j] 
    return padded


def new_conv(img_grb,chw,filters, optim_model, conv, pad=0, stride=1):
    """
    inputs :
        optim_model : the optimization model we use to create the
        img_grb     : an input "image" in the format CHW, a gurobi variable
        chw         : the shape of img_grb
        filters     : the filters in format CHW, a numpy variable, odd number of pixel
        conv        : the variable which will contain the linear expression of the convolution

    output :
        conv : the convolutions written as linear expression that can be used by gurobi

    """

    c,h,w = chw
    print("chw", c, h, w )
    nf,cf, hf, wf = filters.shape
    print( "filters.shape", filters.shape)
    t = hf//2 # we suppose that hf=wf, then we can use the same value to compute where the convolutions starts and where it ends

    #print("t",t)
    #print("h-t",h-t)
    
    print ("cf", cf, "\t c", c)
    if (cf != c) :
        print("the filters has {} channels while the image has {} ".format(cf,c))
        return -1
    
    
    h2 = 1 + ((h-hf+2*pad)/stride)
    w2 = 1 + ((w-wf+2*pad)/stride)
    
    print("h2 {}, w2 {}".format(h2, w2))
    if not ( h2.is_integer() and w2.is_integer()) : 
        print("stride and filter shape are incompatible.")
        return -1
    h2 = int(h2)
    w2 = int(w2)
    #print("h2 : ", h2, "\t w2 : ", w2)
    
    
    #print("img_grb", img_grb.keys())
    #print("filters", filters.shape)
    
    if pad !=0:
        padded_img = grb_padding(img_grb=img_grb, chw=chw, padding=pad)
    else: 
        padded_img = img_grb
    
    for k in range(nf): # on the filters
        for i in range(h2): # on the height
                for j in range(w2): # on the width
                    #print("indices",k, i,j)
                    conv[k,i,j] =  quicksum( [ padded_img[m,i*stride+n,j*stride+u]*filters[k,m,n,u] for m in range(c) for n in range(hf)  for u in range(wf)  ] )
     
                    
    #print("img_grb", img_grb)
    #print("filters", filters)
    return conv






class Wong_Conv_mnist_model(nn.Module): # expected input size 28x28

    def __init__(self, logit_only = False):
        super(Wong_Conv_mnist_model,self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(32*7*7,100)
        self.fc2 = nn.Linear(100, 10) 
        self.logit_only = logit_only

    def forward(self, x):
        bs = x.size()[0] # batch_size
        conv1 = self.conv1(x)
        conv1_relu = F.relu(conv1)
        conv2 = self.conv2(conv1_relu)
        conv2_relu = F.relu(conv2)
        flatten = conv2_relu.view(bs,-1)
        fc1 = self.fc1(flatten)
        fc1_relu = F.relu(fc1)
        output = self.fc2(fc1_relu)
        #output = torch.squeeze(output)
        if self.logit_only : 
            return output
        else :
            return F.softmax(output, dim = -1), output,{'conv1':conv1, 'conv2':conv2, 'fc1':fc1}  # return prob and logit, the "probability" of belonging to one class and the logit used to compute those probabilities, and the result of the convolution


    def num_flat_features(self, x):
        size = x.size() # all dimension of x which is a parameter of the network
        num_features = 1

        for s in size :
            num_features *= s
        return num_features

    def total_num_parameters(self):
        params = list(self.parameters())
        num_params = 0

        for p in params:
            #print(self.num_flat_features(p))
            num_params += self.num_flat_features(p)

        return num_params

    
    
def my_patch_extractor_new(img,hw=5, pad=0, stride=1):
    """
        inputs : 
            img : an image in the format CHW
            hw  : the size of the square patch, and hw is an odd number
        
        output :
            patch_matrix : a matrix containing all the patch, its dimension is (-1, C*hw*hw), and it is arranged channel by channel on each row
                           and the rows (h-hw//2) first rows represents the patch on the first row. Patches are extracted row-wise  

    """

    # getting the dimensions of the image
    #print("imag shape : ", img.shape)
    c,h,w = img.shape
    h2 = 1 + ((h-hw+2*pad)/stride)
    w2 = 1 + ((w-hw+2*pad)/stride)
    if not ( h2.is_integer() and w2.is_integer()) : 
        print("stride and filter shape are incompatible.")
        return -1
    h2 = int(h2)
    w2 = int(w2)    
    
    
    if pad >0:
        img = np.pad(img, ((0,0),(pad,pad), (pad,pad)), 'constant')
        
       
    n_patch = h2*w2
    patch_matrix = np.zeros((n_patch,c*hw*hw))
    
    #print("patch_matrix shape : ", patch_matrix.shape )
    k = 0
    for i in range(h2):
        for j in range(w2):
            #print("patch_shape : ", img[:,i*stride:i*stride+hw,j*stride:j*stride+hw].shape )
            patch_matrix[k,:] = img[:,i*stride:i*stride+hw,j*stride:j*stride+hw].reshape(-1)
            k +=1
            

    #return c,h,w
    return patch_matrix 

def IA_bound_conv_new(filters,bias, lb, ub, patch_size, pad = 0, stride=1) :
    """
    using arithmetic, compute the upper and lower bound of conv(x,filters) where   lb <= x <= ub,
    params : 
        filters : the ndarray of filters;
        bias : bias, the bias term;
        lb : lower bound of x, in CHW format;
        ub : upper bound of x, in CHW format; 
 
    
    returns: 
        upperbound : upper bound of conv(x,filters), where lb <= x <= ub, computed with interval arithmetic
        lowerbound : lower bound of conv(x,filters), where lb <= x <= ub, computed with interval arithmetic
    """
    nf, nc, hf, wf  = filters.shape
    _, h, w = lb.shape
    
    h2 = 1 + ((h-hf+2*pad)/stride)
    w2 = 1 + ((w-wf+2*pad)/stride)
    if not ( h2.is_integer() and w2.is_integer()) : 
        print("stride and filter shape are incompatible.")
        return -1
    h2 = int(h2)
    w2 = int(w2)    
    
    output_shape = [nf,h2, w2]
    
   
    
    filters_2d = filters.reshape(nf,-1)
    f_plus = np.maximum(0, filters_2d)
    f_minus = np.minimum(0,filters_2d)
    
    lb_patch = my_patch_extractor_new(lb,hw=patch_size, pad=pad, stride=stride).T
    ub_patch = my_patch_extractor_new(ub,hw=patch_size, pad=pad, stride=stride).T
    
    #print( "f_plus.shape=", f_plus.shape)
    
    upperbound = f_plus @ ub_patch + f_minus @ lb_patch  
    lowerbound = f_minus @ ub_patch + f_plus @ lb_patch
    
    for i in range(nf):
        upperbound[i,:] += bias[i]
        lowerbound[i,:] += bias[i]
        
    #print("output_shape :", output_shape)
    return lowerbound.reshape(output_shape), upperbound.reshape(output_shape)

def new_conv(img_grb,chw,filters, optim_model, conv, pad=0, stride=1):
    """
    inputs :
        optim_model : the optimization model we use to create the
        img_grb     : an input "image" in the format CHW, a gurobi variable
        chw         : the shape of img_grb
        filters     : the filters in format CHW, a numpy variable, odd number of pixel
        conv        : the variable which will contain the linear expression of the convolution

    output :
        conv : the convolutions written as linear expression that can be used by gurobi

    """

    c,h,w = chw
    #print("chw", c, h, w )
    nf,cf, hf, wf = filters.shape
    #print( "filters.shape", filters.shape)
    
    #print("t",t)
    #print("h-t",h-t)
    
    #print ("cf", cf, "\t c", c)
    if (cf != c) :
        print("the filters has {} channels while the image has {} ".format(cf,c))
        return -1
    
    
    h2 = 1 + ((h-hf+2*pad)/stride)
    w2 = 1 + ((w-wf+2*pad)/stride)
    
    #print("h2 {}, w2 {}".format(h2, w2))
    if not ( h2.is_integer() and w2.is_integer()) : 
        print("stride and filter shape are incompatible.")
        return -1
    h2 = int(h2)
    w2 = int(w2)
    print("h2 : ", h2, "\t w2 : ", w2)
    
    
    #print("img_grb", img_grb.keys())
    #print("filters", filters.shape)
    
    if pad !=0:
        padded_img = grb_padding(img_grb=img_grb, chw=chw, padding=pad)
    else: 
        padded_img = img_grb
    
    for k in range(nf): # on the filters
        for i in range(h2): # on the height
                for j in range(w2): # on the width
                    #print("indices",k, i,j)
                    conv[k,i,j] =  quicksum( [ padded_img[m,i*stride+n,j*stride+u]*filters[k,m,n,u] for m in range(c) for n in range(hf)  for u in range(wf)  ] )
     
                    
    #print("img_grb", img_grb)
    #print("filters", filters)
    return conv