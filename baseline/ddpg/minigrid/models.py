import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def init_weights_biases(size):
    v = 1.0 / np.sqrt(size[0])
    return torch.FloatTensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, n_state, action_num, n_goal, n_hidden1=256, n_hidden2=256, n_hidden3=256, initial_w=3e-3):
        self.n_states = n_state[0]
        # self.n_action = n_action
        self.action_num = action_num  # action_num=3
        self.n_goal = n_goal
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.initial_w = initial_w
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(in_features=self.n_states + self.n_goal, out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1, out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2, out_features=self.n_hidden3)
        self.fc4 = nn.Linear(in_features=self.n_hidden3, out_features=self.action_num)
        # self.output = nn.Linear(in_features=self.n_hidden3,out_features=self.n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = torch.tanh(self.fc4(x))
        # output = F.log_softmax(self.fc4(x),1)
        return output


class Critic(nn.Module):
    def __init__(self, n_state, n_action, n_goal, n_hidden1=256, n_hidden2=256, n_hidden3=256, initial_w=3e-3):
        self.n_state = n_state[0]
        self.n_action = n_action
        self.n_goal = n_goal
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.initial_w = initial_w
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(in_features=self.n_state + self.n_action + self.n_goal, out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1, out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2, out_features=self.n_hidden3)
        self.output = nn.Linear(in_features=self.n_hidden3, out_features=1)

    def forward(self, x, a):
        x = F.relu(self.fc1(torch.cat([x, a], dim=-1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.output(x)

        return output
