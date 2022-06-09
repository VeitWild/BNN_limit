import torch as torch
import matplotlib.pyplot as plt
import numpy as np
from BNN import MLP
from limit_kernel import Kernel_Limit,relu

#Sampling the maringals for the BNN


L = 3

variance_weights = 1*torch.ones(L+2)
variance_bias =1*torch.ones(L+2)
width = torch.cat((torch.ones(1),100*torch.ones(L),torch.ones(1)),0).int()

activation_func = torch.nn.ReLU()

neural_net = MLP(width,activation_fct=activation_func)

def BNN_marginal(x,neural_net,variance_weights,variance_bias,nr_samples):
    M = nr_samples
    f_x = np.zeros(M)
    print(torch.random.initial_seed())

    for m in range(0,M):
        f_x[m] = neural_net(x).detach().numpy()
        neural_net.initialise_weights(weights_variance=variance_weights,bias_variance=variance_bias)
        
    
    return f_x


M= 1000

x=0*torch.ones(1)
f_x = BNN_marginal(x,neural_net,variance_weights,variance_bias,nr_samples=M)

gamma_weights = np.ones(L+2) #first entry irrelevant
gamma_bias = np.ones(L+2) # first entry irrelevant
N_S = 100

k  = Kernel_Limit(nr_hidden_layers=L,gamma_weights=gamma_weights,gamma_bias=gamma_bias,activation_fct=relu,N_S=N_S)


rough_samples = np.zeros((M,2))
smooth_samples = np.zeros((M,2))
x = np.arange(0,2).reshape(2,1)

for m in range(0,M):
   rough_samples[m,] = k.sample_function(x,rough=True).reshape(1,2)
   smooth_samples[m,] = k.sample_function(x,rough=False).reshape(1,2)


#plt.hist(f_x,bins=50,density=True)
#plt.show()

bins = np.linspace(-3, 3, 30)

plt.hist(rough_samples[:,0], bins, alpha=0.5, label='rough GP',density=True)
plt.hist(smooth_samples[:,0], bins, alpha=0.5, label='smooth GP',density=True)
#plt.hist(f_x, bins,alpha=0.5, density=True, label='smooth BNN')

plt.legend(loc='upper right')
plt.show()