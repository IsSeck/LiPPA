
""" utils for cnn architecture """

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
from temp_utils import Wong_Conv_mnist_model, my_coord, new_conv



### class CNN for mnist

class CNN_from_mat(nn.Module): # expected input size 28x28

    def __init__(self, logit_only = False):
        super(CNN_from_mat,self).__init__()
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
        conv2_relu = conv2_relu.permute(0,2,3,1).contiguous() # contiguous added to prevent error from view after permutation
        #import pdb ; pdb.set_trace()
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
##########################

### Converting .mat to .pth

def mat_to_pth(net,mat_path,pth_path=None):
    #import pdb ; pdb.set_trace()
    
    temp_dict = sio.loadmat(mat_path)
    torch_dict = {  'conv1.weight':torch.tensor(temp_dict['weights_conv1']).permute(3,2,0,1),
                    'conv1.bias': torch.tensor(temp_dict['biases_conv1'][0]),
                    'conv2.weight' : torch.tensor(temp_dict['weights_conv2']).permute(3,2,0,1),
                    'conv2.bias' : torch.tensor(temp_dict['biases_conv2'][0]),
                    'fc1.weight' : torch.tensor(temp_dict['weights_fc1'].T),
                    'fc1.bias' :  torch.tensor(temp_dict['biases_fc1'][0]),
                    'fc2.weight' : torch.tensor(temp_dict['weights_fc2'].T),
                    'fc2.bias' : torch.tensor(temp_dict['biases_fc2'][0]) }

    net.load_state_dict(torch_dict)
    if pth_path is not None : 
        torch.save(net.state_dict(),pth_path)

    return net

########################################

### permut to match mmr weights format : 



#############################################

def lp_constructor_cnn(net_dict): 
    """
    Function building the squeleton of the optimisation problem

    input :     + net_dict  : the network parameter dictionary so that we can retrieve needed information we need such as the strides and the padding
                 
    output :    + advex_finder: the root optimisation problem  
                
    
    """

    conv1_w = net_dict['conv1.weight']#.cpu().numpy() # weight
    conv1_b = net_dict['conv1.bias']#.cpu().numpy()   # bias
    conv1_s = net_dict['conv1.stride']                        # stride
    conv1_p = net_dict['conv1.padding']                       # padding

    conv2_w = net_dict['conv2.weight']#.cpu().numpy() 
    conv2_b = net_dict['conv2.bias']#.cpu().numpy()
    conv2_s = net_dict['conv2.stride']
    conv2_p = net_dict['conv2.padding']                       # padding


    fc1_w = net_dict['fc1.weight']#.cpu().numpy() 
    fc1_b = net_dict['fc1.bias']#.cpu().numpy()
    

    fc2_w = net_dict['fc2.weight']#.cpu().numpy() 
    fc2_b = net_dict['fc2.bias']#.cpu().numpy()
    

    # create the model and its variables
    advex_finder = Model("MILP_advex") # creation of the model 

    x_coord = my_coord(1,28,28)
    conv1_coord = my_coord(16,14,14)
    conv2_coord = my_coord(32,7,7)
    fc1_coord = range(100)
    s_coord = range(10)


    epsilon = advex_finder.addVar( lb=0, name="epsilon")  
    x = advex_finder.addVars(x_coord , lb=0 , ub=1 , name = "x")
    sample_grb = advex_finder.addVars(x_coord, lb=0 , ub=1 , name = "sample_grb")

    # conv1
    z1_hat = advex_finder.addVars(conv1_coord, lb = - GRB.INFINITY, ub = GRB.INFINITY, name = "z1_hat" ) # z1_hat, pre-ReLU output of the first conv. layer
    z1 = advex_finder.addVars(conv1_coord, lb = 0, ub = GRB.INFINITY, name = "z1" ) # z1 post-ReLU output of the conv. layer
    b1 = advex_finder.addVars(conv1_coord, name =  "b1")
    

    # conv2
    z2_hat = advex_finder.addVars(conv2_coord, lb = - GRB.INFINITY, ub = GRB.INFINITY, name = "z2_hat" ) 
    z2 = advex_finder.addVars(conv2_coord, lb = 0, ub = GRB.INFINITY, name = "z2" ) #  z = max(0,z_hat), the ReLU
    b2 = advex_finder.addVars(conv2_coord, name =  "b2")

    # fc1 
    z3_hat = advex_finder.addVars(fc1_coord, lb = - GRB.INFINITY, ub = GRB.INFINITY, name = "z3_hat" ) 
    z3 = advex_finder.addVars(fc1_coord, lb = 0, ub = GRB.INFINITY, name = "z3" ) #  z = max(0,z_hat), the ReLU
    b3 = advex_finder.addVars(fc1_coord, name =  "b3")

    # fc2
    s = advex_finder.addVars(s_coord,lb = - GRB.INFINITY, name = "s")



    # infinite norm constraint
    advex_finder.addConstrs( (epsilon >= sample_grb[i]-x[i] for i in x_coord), name = "infinite_norm_ineq1")
    advex_finder.addConstrs( (epsilon >= x[i]-sample_grb[i] for i in x_coord), name = "infinite_norm_ineq2")

    # conv1 + relu
    t1 = new_conv(x,[1,28,28],conv1_w, advex_finder, z1_hat.copy(), stride=conv1_s, pad=conv1_p)
    advex_finder.addConstrs( (z1_hat[i,j,k] - t1[i,j,k] == conv1_b[i] for i in range(16) for j in range(14) for k in range(14) ),"conv1"  )

    # conv2 + relu
    t2 = new_conv(z1,[16,14,14],conv2_w, advex_finder, z2_hat.copy(), stride=conv2_s, pad=conv2_p)
    advex_finder.addConstrs( (z2_hat[i,j,k] - t2[i,j,k] == conv2_b[i] for i in range(32) for j in range( 7) for k in range( 7)),"conv2"  )
    
    # fc1 + relu
    
    #if mmr_permut : 
    #    conv2_coord = [(j,k,i) for i in range(32) for j in range(7) for k in range(7)]
    advex_finder.addConstrs( ( z3_hat[i] - quicksum( [fc1_w[i,j]*z2[conv2_coord[j]] for j in range(1568) ]) == fc1_b[i] for i in range(100)),"fc1"  ) # 1568=32*7*7


    # fc2
    advex_finder.addConstrs( ( s[i] - quicksum( [fc2_w[i,j]*z3[j] for j in range(100) ]) == fc2_b[i] for i in range(10)),"fc2"  )
    
    advex_finder.setObjective(epsilon,GRB.MINIMIZE)
    advex_finder.update()


    return advex_finder


#############################

def my_get_var_3d(optimodel, variable_name, shape ): # fonction pour la récupération des variables du model
    x = np.zeros(shape)
    c,h,w = shape
    
    for i in range(c):
        for j in range(h):
            for k in range(w):
                varname = variable_name + "[{},{},{}]".format(i,j,k)
                x[i,j,k] = optimodel.getVarByName(varname).X
    
    return x

#########################

def linear_region_diff(z_hat_x1,z_hat_x2):
    """
    given the activations(before ReLU) of two samples, compute the number of activation differing from them
    """

    b1_x1 = (z_hat_x1['conv1'][0]>=0).cpu().numpy()
    b2_x1 = (z_hat_x1['conv2'][0]>=0).cpu().numpy()
    b3_x1 = (z_hat_x1['fc1'][0]>=0).cpu().numpy()

    b1_x2 = (z_hat_x2['conv1'][0]>=0).cpu().numpy()
    b2_x2 = (z_hat_x2['conv2'][0]>=0).cpu().numpy()
    b3_x2 = (z_hat_x2['fc1'][0]>=0).cpu().numpy()
    

    conv1_diff = np.sum(b1_x1!=b1_x2)
    conv2_diff = np.sum(b2_x1!=b2_x2)
    fc1_diff = np.sum(b3_x1!=b3_x2)
    total_diff = conv1_diff + conv2_diff + fc1_diff 
    return {'total_diff' : total_diff, 'conv1_diff' : conv1_diff, 'conv2_diff' :  conv2_diff, 'fc1_diff' : fc1_diff}


#########################
def lp_cnn(root_advex_finder, sample_np, activation,pred_dict,tolerance,device='cpu', verbose=False):
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
    
    lp_advex_finder = root_advex_finder.copy()

    y_np = pred_dict['y_np']
    y_2nd_np =  pred_dict['y_2nd_np']
    
    x_coord = my_coord(1,28,28)
    conv1_coord = my_coord(16,14,14)
    conv2_coord = my_coord(32,7,7)
    fc1_coord = range(100)
    s_coord = range(10)

    b1 = (activation['conv1'][0]>=0).cpu().numpy()+0.0
    b2 = (activation['conv2'][0]>=0).cpu().numpy()+0.0
    b3 = (activation['fc1'][0]>=0).cpu().numpy()+0.0



    m = 1e-6 # to prevent adversarial examples from exiting the linear region

    #print("b1.shape ", b1.shape, "\n b2.shape ", b2.shape)
    
    # sample_grb
    for i in x_coord:
        i0,i1,i2 = i
        lp_advex_finder.addConstr( ( lp_advex_finder.getVarByName("sample_grb[{},{},{}]".format(i0,i1,i2))==sample_np[i0,i1,i2] ), "sample_constrs_{}_{}_{}".format(i0,i1,i2) )
    
    # conv1 
    for i in conv1_coord :
        i0,i1,i2 = i
        z1_hat_temp = lp_advex_finder.getVarByName('z1_hat[{},{},{}]'.format(i0,i1,i2))
        z1_temp = lp_advex_finder.getVarByName('z1[{},{},{}]'.format(i0,i1,i2))
        #print(type(z1_hat_temp), type(z1_temp))
        #import pdb ; pdb.set_trace()
        if b1[i]==0.0 : 
            lp_advex_finder.addConstr(z1_hat_temp<=-m, 'conv1_inactive_1o2_{}'.format(i))
            lp_advex_finder.addConstr(z1_temp==0, 'conv1_inactive_2o2_{}'.format(i))
        if b1[i]==1.0 : 
            lp_advex_finder.addConstr(z1_hat_temp>=m, 'conv1_active_1o2_{}'.format(i))
            lp_advex_finder.addConstr(z1_temp==z1_hat_temp, 'conv1_active_2o2_{}'.format(i))


    # conv2
    for i in conv2_coord :
        i0,i1,i2 = i
        z2_hat_temp = lp_advex_finder.getVarByName('z2_hat[{},{},{}]'.format(i0,i1,i2))
        z2_temp = lp_advex_finder.getVarByName('z2[{},{},{}]'.format(i0,i1,i2))
        if b2[i]==0.0 : 
            lp_advex_finder.addConstr(z2_hat_temp<=-m, 'conv2_inactive_1o2_{}'.format(i))
            lp_advex_finder.addConstr(z2_temp==0, 'conv2_inactive_2o2_{}'.format(i))
        if b2[i]==1.0 : 
            lp_advex_finder.addConstr(z2_hat_temp>=m, 'conv2_active_1o2_{}'.format(i))
            lp_advex_finder.addConstr(z2_temp==z2_hat_temp, 'conv2_active_2o2_{}'.format(i))
        

    for i in fc1_coord : 
        z3_hat_temp = lp_advex_finder.getVarByName('z3_hat[{}]'.format(i))
        z3_temp = lp_advex_finder.getVarByName('z3[{}]'.format(i))
        if b3[i]==0.0 : 
            lp_advex_finder.addConstr(z3_hat_temp<=-m, 'fc1_inactive_1o2_{}'.format(i))
            lp_advex_finder.addConstr(z3_temp==0, 'fc1_inactive_2o2_{}'.format(i))
        if b3[i]==1.0 : 
            lp_advex_finder.addConstr(z3_hat_temp>=m, 'conv2_active_1o2_{}'.format(i))
            lp_advex_finder.addConstr(z3_temp==z3_hat_temp, 'conv2_active_2o2_{}'.format(i))
        

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
    
        adv_ex_lp = my_get_var_3d(optimodel=lp_advex_finder, variable_name='x', shape=(1,28,28) )#my_get_var(lp_advex_finder,"x")
        adv_ex_lp = torch.tensor(adv_ex_lp.reshape(1,28,28)).to(device).type(torch.float32)
    
    return eps_final, adv_ex_lp, lp_advex_finder



######


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
    sample_np = sample[0].detach().cpu().numpy()
    
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
                x_adv = my_fgsm(net,eps_fgsm,x_adv.view(1,1,28,28).detach(),torch.tensor(true_label), twister = 1) # compute an adversarial example using fgsm
                _, _, z_hat_adv  = net(x_adv)
                fgsm_it += 1 

            
    return None, None
