from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

def init_weights_biases(size):
    v = 1.0/np.sqrt(size[0])
    return torch.FloatTensor(size).uniform_(-v,v)

class Actor(nn.Module):
    def __init__(self,n_state,n_hidden1=400,n_hidden2=300,initial_w = 3e-3):
        self.n_states = n_state[0]
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.initial_w = initial_w
        super(Actor,self).__init__()

        self.fc1 = nn.Linear(in_features=self.n_states,out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1,out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2,out_features=1)
        self.tanh = nn.Tanh()

        self.fc1.weight.data = init_weights_biases(self.fc1.weight.data.size())
        self.fc1.bias.data = init_weights_biases(self.fc1.bias.data.size())
        self.fc2.weight.data = init_weights_biases(self.fc2.weight.data.size())
        self.fc2.bias.data = init_weights_biases(self.fc2.bias.data.size())
        self.fc3.weight.data.uniform_(-self.initial_w,self.initial_w)
        self.fc3.bias.data.uniform_(-self.initial_w,self.initial_w)
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.tanh(self.fc3(x))
class Critic(nn.Module):
    def __init__(self,n_state,n_action,n_hidden1=400,n_hidden2=300,initial_w = 3e-3):
        self.n_states = n_state[0]
        self.n_action = n_action
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.initial_w = initial_w
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(in_features=self.n_states+self.n_action,out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1,out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2,out_features=1)

        self.fc1.weight.data = init_weights_biases(self.fc1.weight.data.size())
        self.fc1.bias.data = init_weights_biases(self.fc1.bias.data.size())
        self.fc2.weight.data = init_weights_biases(self.fc2.weight.data.size())
        self.fc2.bias.data = init_weights_biases(self.fc2.bias.data.size())
        self.fc3.weight.data.uniform_(-self.initial_w,self.initial_w)
        self.fc3.bias.data.uniform_(-self.initial_w,self.initial_w)
    def forward(self,state,action):
        x = F.relu(self.fc1(torch.cat([state.float(),action.float()],dim=1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)





