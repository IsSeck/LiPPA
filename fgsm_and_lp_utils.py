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
import time
import csv





def algo1_mlp256(sample, net,bound_dict,pred_dict, eps_fgsm ,tolerance, optim_time = 10, device = 'cpu' ) : 
    """ 
    input : 
            + sample : (torch) the sample around which we are looking for adversarial examples
            + net : the network 
            + bound_dict : a dictionary containing the bounds on the activations before the ReLU
            + pred_dict : a dictionary containing the class predicted 
    output : 
            + output_dict1 : dictionary containing values to store in a csv file
            + output_dict2 : dictionary containing values to plot
    """
    
    tic_global = time.time()
    
    sample_np = sample.detach().view(-1).cpu().numpy()
    net_params = net.state_dict()
    net_dict = { key : value.cpu().numpy() for key,value in zip(net_params.keys(),net_params.values()) }
    root_advex_finder = lp_constructor_mlp256(sample_np, net_dict, bound_dict, pred_dict,tolerance)
    fgsm_it_list = []
    eps_lp_list = []
    
    total_it = 0
    number_of_restart = 0
    restart_index = []
    
    sample = sample.to(device)
    net = net.to(device)
    _, _, z_hat_ref  = net(sample, prob_only=False)
    binary_ref = (z_hat_ref >=0)[0].cpu().numpy() 
    eps_ref, x_temp, _ = lp_mlp256(root_advex_finder,binary_ref)
    
    if eps_ref is not None :
        eps_h = eps_ref
        b_h = binary_ref
        x_h = x_temp
        index_b_h = 0
    else :
        eps_h = 10
    
    output_dict_1 = {'eps_ref':eps_ref} # dictionary that is going to be written in  a csv
    output_dict_2 = {} # dictionary that will eventually be store or used to plot 
    x_adv = sample
    label = torch.tensor(pred_dict['y_np']).to(device)
    
    binary = binary_ref
    tic_while = time.time()
    while time.time()-tic_while < optim_time : 
        
        binary_temp = binary
        fgsm_it = 0
        while sum(binary!=binary_temp) == 0 or fgsm_it<10  : 
            x_adv = my_fgsm(net,eps_fgsm,x_adv,label, twister = 1) # compute an adversarial example using fgsm
            _, _, z_hat_adv  = net(x_adv, prob_only=False)
            binary = (z_hat_adv >=0)[0].cpu().numpy() # these lines are used to get the activation for the adv_ex
            fgsm_it += 1 
        fgsm_it_list.append(fgsm_it)
            
            
        eps_lp, x_adv, _ = lp_mlp256(root_advex_finder,binary,device)
        eps_lp_list.append(eps_lp)
        total_it += 1
        
        if eps_lp < eps_h :
            eps_h = eps_lp
            b_h = binary
            index_b_h = total_it
            x_h = x_adv#.detach().cpu().numpy()
        
        if len(eps_lp_list)>4 and abs(eps_lp_list[-1]-eps_lp_list[-3])<1e-5 :
            x_adv = sample.detach().to(device) + eps_h * (2*torch.rand((1,28,28))-1).to(device)
            x_adv = torch.clamp(x_adv, min=0, max=1)
            number_of_restart += 1
            restart_index.append(total_it)
            #stop = True
        
        fgsm_avg_it = np.mean(fgsm_it_list)
        
        
        
    output_dict_1['eps_h'] = eps_h
    output_dict_1['number_of_binary_modifications'] = sum(b_h!=binary_ref)
    output_dict_1['number_of_restart'] = number_of_restart
    output_dict_1['total_it'] = total_it
    output_dict_1['fgsm_avg_it'] = fgsm_avg_it
    output_dict_1['runtime'] = time.time()-tic_global
    output_dict_1['optim_time'] = optim_time
    
    
    
    output_dict_2['x_h'] = x_h
    output_dict_2['b_h'] = b_h
    output_dict_2['index_b_h'] = index_b_h
    output_dict_2['eps_lp_list'] = eps_lp_list
    output_dict_2['fgsm_it_list'] = fgsm_it_list
    output_dict_2['restart_index'] = restart_index

    return output_dict_1, output_dict_2
    
    
def my_get_var(optimodel, variable_name, size = 784): # fonction pour la récupération des variables du model
    x = np.zeros(size)
    for i in range(size):
        varname = variable_name + "[{}]".format(i)
        x[i] = optimodel.getVarByName(varname).X
    
    return x


def lp_mlp256(root_advex_finder, activation,device='cpu'):
    """
    input : 
            + root_advex_finder : lp constructed waiting for the value of "binary" activations
            + activation : activations that fix the "binary" activations in root_advex_finder making it a linear program
    output : 
            + eps_final : the value of epsilon found for activation
            + x_adv : the adversarial example achieving that eps_final (in torch)
            + lp_advex_finder : the optimization problem.
    
    """
    
    lp_advex_finder = root_advex_finder.copy()
    lp_advex_finder.addConstrs( ( lp_advex_finder.getVarByName("b[{}]".format(i))==activation[i] for i in range(e)), "lp_activation_constrs" )  
    #lp_advex_finder.Params.outputflag = 1
    lp_advex_finder.optimize()

    eps_final = None
    adv_ex_lp = None
    if lp_advex_finder.solcount != 0 :
        eps_final = lp_advex_finder.Objval
    
        adv_ex_lp = my_get_var(lp_advex_finder,"x")
        adv_ex_lp = torch.tensor(adv_ex_lp.reshape(1,28,28)).to(device).type(torch.float32)
    
    return eps_final, adv_ex_lp, lp_advex_finder
    
    
    
def lp_constructor_mlp256(sample_np, net_dict, bound_dict,pred_dict,  tolerance = 0.0001, verbose = False):
    """
    input :     + sample : a numpy sample (or a batch of only one image)
                + net_dict : the network params dictionary so that we can retrieve all the information we need
                + bound_dict: dictionary containing the bounds of activation for each neuron
                + activation : the activation used for our LP
                
    output :    + eps_0 : the closest adversarial example if it exist or None
                + output: the output of the neural network
                + advex : the adversarial example in a vector
                + runtime : the runtime 
                
    
    """
    
    #print("predicted class : {} \t 2nd class : {}".format(y_np,y_2nd_np))
    #desired_output = [i for i in range(10) if i!=y_np]
    
    W = net_dict['fc1.weight']   # shape =  e x d
    beta = net_dict['fc1.bias']  # shape = e
    V = net_dict['fc2.weight']   # shape = c x e
    alpha = net_dict['fc2.bias'] # shape = c
    

    
    #sample_np = sample.view(-1).cpu().numpy()
    
    # compute the upper and lower bounds for the Big_M formulations
    M_lb,M_ub = bound_dict['M_lb'],bound_dict['M_ub'] 
    
    y_np = pred_dict['y_np']
    y_2nd_np =  pred_dict['y_2nd_np']
    
    advex_finder = Model("LP_advex_for_dca")

    epsilon = advex_finder.addVar(  name="epsilon")
    b = advex_finder.addVars(range(e), name =  "b")
    #for i in I : b[i].start = b_anchor[i] # warm-start
    x = advex_finder.addVars(range(d), lb=0 , ub=1 , name = "x")
    z_hat = advex_finder.addVars(range(e), lb = M_lb, ub =M_ub, name = "z_hat" ) # never forget 
    z = advex_finder.addVars(range(e), name = "z" )
    s = advex_finder.addVars(range(c),lb = -GRB.INFINITY, name = "s")
    #bs = advex_finder.addVars(desired_output, vtype = GRB.BINARY, name =  "bs")
    #bs[y_2nd_np].start =1
    winning = advex_finder.addVar(lb = -GRB.INFINITY, name = "winning")
    # infinite norm constraint
    #for i in range(d):
    advex_finder.addConstrs( (epsilon >= sample_np[i]-x[i] for i in range(d)), name = "infinite_norm_1o2" )
    advex_finder.addConstrs( (epsilon >= x[i]-sample_np[i] for i in range(d)), name = "infinite_norm_2o2" )

    # constraints z_hat = W x + beta
    #for i in range(e):
    advex_finder.addConstrs(( z_hat[i] - quicksum( [W[i,j]*x[j] for j in range(d) ]) == beta[i] for i in range(e)),"perceptron"  )

    #advex_finder.addConstr( z[i]  == max_(0,z_hat[i]), "ReLU_1%d" % i )
    advex_finder.addConstrs( (z[i] >= 0 for i in range(e)), "relu_positive"  )
    advex_finder.addConstrs( (z[i] >= z_hat[i] for i in range(e)), "relu_greater_than" )
    advex_finder.addConstrs( (z[i] <= abs(M_ub[i])*b[i] for i in range(e)), "relu_ub" )
    advex_finder.addConstrs( (z[i] <= z_hat[i] + abs(M_lb[i])*(1-b[i]) for i in range(e)) ,"relu_lb"   )


    
    advex_finder.addConstrs( (s[i]- quicksum( [V[i,j]*z[j] for j in range(e) ]) == alpha[i] for i in range(c) ),"output" )

    advex_finder.addConstr(s[y_2nd_np]-s[y_np]>=tolerance, "adversarial"  )
    

    #advex_finder.Params.TimeLimit = 60 #1200
    advex_finder.setObjective(epsilon,GRB.MINIMIZE)
    advex_finder.Params.outputflag = verbose

    advex_finder.update()
    return advex_finder
    
    
    
def my_cross_entropy(y,y_pred): # computing the loss for a single element
    exps = torch.exp(torch.squeeze(y_pred))
    y = y.type(torch.LongTensor)
    exps = exps/torch.sum(exps) # normalize to have probabilities
    #print(exps)
    return -torch.log(exps)[y], exps


def my_fgsm(model, eps, images, labels, twister = 1,device='cpu'): #https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/fgsm.html#FGSM
        
        model.to(device)
        images = images.to(device)
        labels = labels.to(device)
        #loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        #outputs = model(images, prob_only=True) #  see https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
        prob , logit, _ = model(images)
        #print(outputs)
        cost,_ = my_cross_entropy(labels,logit) #loss(outputs, labels).to(device)
        #print(cost)
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + twister * eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images


    
    

class mlp_16_16(nn.Module): # expected input size 28x28

    def __init__(self, logit_only=False):
        super(mlp_16_16,self).__init__()
        self.fc1 = nn.Linear(784,16) # an affine operation :y = Wx + b
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,10)
        self.logit_only = logit_only

        
    def forward(self, x, logit_only = False):
        bs = x.size()[0] # batch_size
        x = x.view(bs,-1) # flattening the output for the fully connected layer.
        fc1 = self.fc1(x)
        fc1_relu = F.relu(fc1)
        fc2 = self.fc2(fc1_relu)
        fc2_relu = F.relu(fc2)
        output = self.fc3(fc2_relu)
        #output = torch.squeeze(output)
        if self.logit_only :
            return output
        else :
            return F.softmax(output, dim = -1), output, {"fc1":fc1,"fc2":fc2} # return prob and logit, the "probability" of belonging to one class and the logit used to compute those probabilities


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

