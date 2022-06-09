import torch as torch
import matplotlib.pyplot as plt

class MLP(torch.nn.Module):
    def __init__(self,width,activation_fct):
        super(MLP,self).__init__()

        self.width = width
        L = width.shape[0]-2 # width[0],...,width[L+1]
        self.L = L

        for l in range(1,L+1):
            setattr(self,'linear'+str(l),torch.nn.Linear(width[l-1],width[l]))
            setattr(self,'activation'+str(l),activation_fct)
            
        self.out = torch.nn.Linear(width[L], width[L+1])
    
    def forward(self,x):
        L = self.width.shape[0]-2

        for l in range(1,L+1):
            x = getattr(self,'linear'+str(l))(x)
            x = getattr(self,'activation'+str(l))(x)
        
        x = self.out(x)

        return x
    
    def initialise_weights(self,weights_variance,bias_variance):

        L = self.L
        D = self.width

        weights_sample_list = [None] * (L+4)
        bias_sample_list = [None] * (L+4)

        for l in range(1,L+2):
            weights_sample_list[l] = torch.sqrt(weights_variance[l]/D[l-1]) *torch.normal(0,1 , size=(D[l],D[l-1])) #  W^L \sim N(0, gamma_w/D_l-1)
            bias_sample_list[l] = torch.sqrt(bias_variance[l]/D[l-1]) *torch.normal(0,1 , size=(D[l],)) #  b^l \sim N(0, gamma_b/D_l-1)

        l=1
        #for name, param in self.named_parameters():
        #    param.data = combined_list[l]
            
        for layer in self.children():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data = weights_sample_list[l]
                layer.bias.data = bias_sample_list[l]
                #print(l)
                l = l+1
        
        
L = 1

variance_weights = 1*torch.ones(L+2)
variance_bias =1*torch.ones(L+2)
width = torch.cat((torch.ones(1),50*torch.ones(L),torch.ones(1)),0).int()

activation_func = torch.nn.ReLU()

neural_net = MLP(width,activation_fct=activation_func)

N= 100
X =torch.linspace(-3,3,N).reshape(N,1)

print(neural_net)
neural_net.initialise_weights(weights_variance=variance_weights,bias_variance=variance_bias)
f_X = neural_net(X)


#plt.plot(X,f_X.detach().numpy())
#plt.show()