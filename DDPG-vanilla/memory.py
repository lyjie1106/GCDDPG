import random
from collections import namedtuple

import torch

Transition = namedtuple('transition',('state','reward','done','action','next_state'))

class Memory:
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.memory_counter = 0

    def add(self,*transition):
        self.memory.append(Transition(*transition))
        if len(self.memory)>self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def unpack_batch(batch,batch_size,n_state,n_action):
        batch = Transition(*zip(*batch))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        states = torch.cat(batch.state).to(device).view(batch_size,*n_state)
        actions = torch.cat(batch.action).to(device).view((-1,1))
        rewards = torch.cat(batch.reward).to(device).view(batch_size,1)
        dones = torch.cat(batch.done).to(device).view(batch_size,1)
        next_states = torch.cat(batch.next_state).to(device).view(batch_size,*n_state)
        return states,actions,rewards,dones,next_states