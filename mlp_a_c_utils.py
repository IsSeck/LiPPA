""" utils for mlp_a and mlp_c architecture """

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


class one_hidden_layer(nn.Module): # expected input size 28x28
        
    def __init__(self, num_neurons, logit_only=False):
        super(one_hidden_layer,self).__init__()
        self.fc1 = nn.Linear(784,num_neurons) # an affine operation :y = Wx + b
        self.fc2 = nn.Linear(num_neurons,10)
        self.logit_only = logit_only
        self.num_neurons = num_neurons
        
    def forward(self, x):
        bs = x.size()[0] # batch_size
        x = x.view(bs,-1) # flattening the output for the fully connected layer.
        fc1 = self.fc1(x)
        fc1_relu = F.relu(fc1)
        output = self.fc2(fc1_relu)
        if self.logit_only :
            return output
        else :
            return F.softmax(output, dim = -1), output, {"fc1":fc1} # return prob and logit, the "probability" of belonging to one class and the logit used to compute those probabilities


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

###########################

def mat_to_pth(net,mat_path,pth_path=None):
    temp_dict = sio.loadmat(mat_path)
    torch_dict = {  'fc1.weight' : torch.tensor(temp_dict['U'].T),
                    'fc1.bias' :  torch.tensor(temp_dict['bU'][0]),
                    'fc2.weight' : torch.tensor(temp_dict['W'].T),
                    'fc2.bias' : torch.tensor(temp_dict['bW'][0]) }

    net.load_state_dict(torch_dict)
    if pth_path is not None : 
        torch.save(net.state_dict(),pth_path)

    return net
##################################


def lp_constructor_one_hidden_layer(net_dict): 
    """
    Function building the squeleton of the optimisation problem

    input :     + net_dict  : the network params dictionary so that we can retrieve all the information we need
                
    output :    + advex_finder: the root optimisation problem  
                
    
    """
    
    W1 = net_dict['fc1.weight']   
    bias1 = net_dict['fc1.bias']     
    W2 = net_dict['fc2.weight']  
    bias2 = net_dict['fc2.bias']     
    
    num_neurons = W1.shape[0]
    advex_finder = Model("LP_advex")

    epsilon = advex_finder.addVar(  name="epsilon")
    b1 = advex_finder.addVars(range(num_neurons), name =  "b1")
    x = advex_finder.addVars(range(784), lb=0 , ub=1 , name = "x")
    sample_grb = advex_finder.addVars(range(784), lb=0 , ub=1 , name = "sample_grb")
    z1_hat = advex_finder.addVars(range(num_neurons), lb = -100, ub =100, name = "z1_hat" ) # never forget 
    z1 = advex_finder.addVars(range(num_neurons), name = "z1" )
    s = advex_finder.addVars(range(10),lb = -GRB.INFINITY, name = "s")

    # infinite norm constraint
    advex_finder.addConstrs( (epsilon >= sample_grb[i]-x[i] for i in range(784)), name = "infinite_norm_1o2" )
    advex_finder.addConstrs( (epsilon >= x[i]-sample_grb[i] for i in range(784)), name = "infinite_norm_2o2" )

    # constraints z1_hat = W1 x + b1 and ReLU
    advex_finder.addConstrs(( z1_hat[i] - quicksum( [W1[i,j]*x[j] for j in range(784) ]) == bias1[i] for i in range(num_neurons)),"layer1"  )
    """
    advex_finder.addConstrs( (z1[i] >= 0 for i in range(100)), "l1_relu_positive"  )
    advex_finder.addConstrs( (z1[i] >= z1_hat[i] for i in range(100)), "l1_relu_greater_than" )
    advex_finder.addConstrs( (z1[i] <= M*b1[i] for i in range(100)), "l1_relu_ub" )
    advex_finder.addConstrs( (z1[i] <= z1_hat[i] + M*(1-b1[i]) for i in range(100)) ,"l1_relu_lb"   )
    """
    
    advex_finder.addConstrs(( s[i] - quicksum( [W2[i,j]*z1[j] for j in range(num_neurons) ]) == bias2[i] for i in range(10)),"output"  )
    
    advex_finder.setObjective(epsilon,GRB.MINIMIZE)
    advex_finder.update()
    
    return advex_finder


#############


def lp_one_hidden_layer(root_advex_finder, sample_np, activation,pred_dict,tolerance,device='cpu', verbose=False):
    """
    Function building and solving the problem on a particular linear region defined by the activations. 

    input : + root_advex_finder : lp constructed waiting for the value of "binary" activations
            + sample : the sample around which adversarial example are searched
            + activation : dictionary containing the output of the two hidden layers (before ReLU)
            + tolerance : for the constraint s[y_2nd_np]-s[y]>=tolerance used for our LP
            + device : choose the device for torch for the adversarial example found 
    
    output : 
            + eps_final : the value of epsilon found for activation
            + x_adv : the adversarial example achieving that eps_final (in torch)
            + lp_advex_finder : the optimization problem.
    
    """
    
    y_np = pred_dict['y_np']
    y_2nd_np =  pred_dict['y_2nd_np']
    
    b1 = (activation['fc1'][0]>=0).cpu().numpy()+0.0
    num_neurons = activation['fc1'][0].shape[0]
    #print("activation nulle ", np.where(activation['fc1'][0]==0), np.where(activation['fc2'][0]==0) )
  
    lp_advex_finder = root_advex_finder.copy()
    lp_advex_finder.addConstrs( ( lp_advex_finder.getVarByName("sample_grb[{}]".format(i))==sample_np[i] for i in range(784)), "sample_constrs" )
    m = 1e-6

    for i in range(num_neurons) :
        z1_hat_temp = lp_advex_finder.getVarByName('z1_hat[{}]'.format(i))
        z1_temp = lp_advex_finder.getVarByName('z1[{}]'.format(i))
        if b1[i]==0.0 : 
            lp_advex_finder.addConstr(z1_hat_temp<=-m, 'l1_inactive_1o2_{}'.format(i))
            lp_advex_finder.addConstr(z1_temp==0, 'l1_inactive_2o2_{}'.format(i))
        if b1[i]==1.0 : 
            lp_advex_finder.addConstr(z1_hat_temp>=m, 'l1_active_1o2_{}'.format(i))
            lp_advex_finder.addConstr(z1_temp==z1_hat_temp, 'l1_active_2o2_{}'.format(i))

    
    s_target = lp_advex_finder.getVarByName("s[{}]".format(y_2nd_np))
    s_origin = lp_advex_finder.getVarByName("s[{}]".format(y_np))
    lp_advex_finder.addConstr(   s_target- s_origin>=tolerance, "adversarial"  )

    
    lp_advex_finder.Params.outputflag = verbose
    lp_advex_finder.Params.numericfocus = 0 # can be setted to 0, does not affect the difference between the s found here and the logit found by torch for the same input
    #lp_advex_finder.Params.presolve = 2
    #lp_advex_finder.Params.intfeastol = 1e-9 # only affects MIP
    lp_advex_finder.optimize()


    eps_final = None
    adv_ex_lp = None
    if lp_advex_finder.solcount != 0 :
        eps_final = lp_advex_finder.Objval
    
        adv_ex_lp = my_get_var(lp_advex_finder,"x")
        adv_ex_lp = torch.tensor(adv_ex_lp.reshape(1,28,28)).to(device).type(torch.float32)
    
    return eps_final, adv_ex_lp, lp_advex_finder

#######

def linear_region_diff(z_hat_x1,z_hat_x2):
    """
    given the activations(before ReLU) of two samples, compute the number of activation differing from them
    """
    b1_x1 = (z_hat_x1['fc1'][0]>=0).cpu().numpy()
    
    b1_x2 = (z_hat_x2['fc1'][0]>=0).cpu().numpy()
    
    layer1_diff = sum(b1_x1!=b1_x2)
    total_diff = layer1_diff 
    return {'total_diff' : total_diff}

####
def lippa(sample, root_advex_finder, lp_method, net,pred_dict, eps_fgsm ,tolerance, optim_time = 10, device = 'cpu',verbose = False, eps_h = 0.1):
    """ 
    input : 
            + sample : (torch) the sample around which we are looking for adversarial examples
            + root_advex_finder : the squeleton of the problem
            + net : the network 
            + pred_dict : a dictionary containing the class the true class and the target class
            + eps_fgsm : the value of the step to use when following the gradient to change linear region
            + tolerance : the s_target-s_true_label>=tolerance
            + optim_time : maximum allowed time for optimization
            + device : the device to use for pytorch computation 
            + verbose : True if you want to see the gurobi print 
            + eps_h : the linf distance in which we look for adversarial examples
    output : 
            + eps, x_adv  : a value of epsilon <= eps_h and the input associated with that or None and None
    """
    
    
    fgsm_it_list = []
    eps_lp_list = []
    
    total_it = 0
    number_of_restart = 0
    restart_index = []
    
    sample = sample.to(device)
    sample_np = sample.detach().view(-1).cpu().numpy()
    
    net = net.to(device)
    _, logit_ref, z_hat_ref  = net(sample)
    z_hat_adv = z_hat_ref # initialize for the while loop
    true_label = pred_dict['y_np'] # we call this function only when pred_dict['y_np']==true_label
    # search the original linear region
    eps_ref, x_temp, _ = lp_method(root_advex_finder,sample_np, z_hat_ref,pred_dict,tolerance,device, verbose)

        # if solution found then stop
    if eps_ref is not None and eps_ref <eps_h :  
        return eps_ref, x_temp

        # else try pgd_restart :
    else : 
        success, x_adv = pgd_restart(net, sample, true_label, eps_h, root_advex_finder ,  num_eps_increase = 5, pgd_increase_rate = 1.05, num_random_start = 50, tolerance = 0.01)
            # if solution found then stop
        linf = torch.max(torch.abs(x_adv-sample)).detach().cpu().numpy()
        if success and linf<=eps_h : 
            return linf, x_adv
        elif success : 
            _, _, z_hat_adv  = net(x_adv)
            # else: 
                # use the solution found as initialisation point for the following

    # while stopping condition :
    tic_while = time.time()
    while time.time()-tic_while < optim_time : # or number_of_restart<3 : # 2 restart should be enough
        eps_lp, x_adv, lp = lp_method(root_advex_finder, sample_np, z_hat_adv,pred_dict,tolerance,device='cpu', verbose=False)
        total_it += 1
        if eps_lp is not None : eps_lp_list.append(eps_lp)
        
        #  if solution was found then stop
        if eps_lp is not None and eps_lp <eps_h :
            return eps_lp, x_adv
        # else :
        else : 
            # bim to change linear region
            if (len(eps_lp_list)>4 and abs(eps_lp_list[-1]-eps_lp_list[-3])<1e-4) or (eps_lp is None) or total_it%20==0 :
                success, x_adv = pgd_restart(net, sample, true_label, eps_h, root_advex_finder ,  num_eps_increase = 5, pgd_increase_rate = 1.05, num_random_start = 50, tolerance = 0.01)
                number_of_restart += 1
                # if solution found then stop
                linf = torch.max(torch.abs(x_adv-sample)).detach().cpu().numpy()
                if success and linf<=eps_h : 
                    return linf, x_adv
                elif x_adv is None : 
                    x_adv = sample.detach().to(device) + eps_h * (2*torch.rand((1,28,28))-1).to(device)
                    x_adv = torch.clamp(x_adv, min=0, max=1)
                    _, _, z_hat_adv  = net(x_adv)
                else  : # else an adversarial example is found but outside of the desire the region
                    _, _, z_hat_adv  = net(x_adv)


                        
            
            z_hat_temp = z_hat_adv
            fgsm_it = 0
            while linear_region_diff(z_hat_adv,z_hat_temp)['total_diff'] == 0 and fgsm_it<50 : 
                x_adv = my_fgsm(net,eps_fgsm,x_adv.detach(),torch.tensor(true_label), twister = 1) # compute an adversarial example using fgsm
                _, _, z_hat_adv  = net(x_adv)
                fgsm_it += 1 


        



            
    return None, None
