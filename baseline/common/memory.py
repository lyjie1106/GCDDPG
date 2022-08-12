import numpy as np

from baseline.common.cher import CHER_sampler
from baseline.common.her import HER_sampler
from baseline.common.vanilla_sampler import Vanilla_sampler


class Memory:
    def __init__(self, env, config):
        self.capacity = config['MEMORY_CAPACITY']
        self.memory = []
        self.memory_counter = 0
        self.memory_length = 0
        self.env = env
        self.Sampler_type = config['Sampler']
        if self.Sampler_type == 'CHER':
            self.sampler = CHER_sampler(config)
        elif self.Sampler_type == 'HER':
            self.sampler = HER_sampler(config)
        elif self.Sampler_type == 'Vanilla':
            self.sampler = Vanilla_sampler(config)

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

    def sample_for_normalization(self, batches):
        size = len(batches[0]['next_state'])
        states, desired_goals = self.sampler.sample_for_normalization(batches, size)
        return self.clip_obs(states), self.clip_obs(desired_goals)

    def sample(self, batch_size):
        states, actions, rewards, next_states, desired_goals = self.sampler.sample(self.memory, batch_size,
                                                                                   self.env.compute_reward)
        return self.clip_obs(states), actions, rewards, self.clip_obs(next_states), self.clip_obs(desired_goals)
