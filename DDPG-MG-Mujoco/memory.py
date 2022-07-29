import numpy as np
from copy import deepcopy
from her import HER_sampler
from cher import CHER_sampler

class Memory:
    def __init__(self, capacity, k_future, env):
        self.capacity = capacity
        self.memory = []
        self.memory_counter = 0
        self.memory_length = 0
        self.env = env
       # self.her = HER_sampler(k_future)
        self.her = CHER_sampler(k_future)

    def __len__(self):
        return len(self.memory)

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    @staticmethod
    def clip_obs(x):
        return np.clip(x, -200, 200)

    def sample_for_normalization(self, batchs):
        size = len(batchs[0]['next_state'])
        states, desired_goals = self.her.sample_for_normalization(batchs, size)
        return self.clip_obs(states), self.clip_obs(desired_goals)

    def sample(self, batch_size):
        states, actions, rewards, next_states, desired_goals = self.her.sample(self.memory, batch_size,
                                                                               self.env.compute_reward)
        return self.clip_obs(states), actions, rewards, self.clip_obs(next_states), self.clip_obs(desired_goals)
