import numpy as np
from copy import deepcopy

from baseline.common.her_ebp import HER_Energy_sampler

G = 9.81
m = 1
Delta_t = 0.04
weight_potential = 1.0
weight_kinetic  = 1.0
max_energy = 999


class MemoryEnergy:
    def __init__(self, capacity, k_future, env, env_name):
        self.capacity = capacity
        self.memory = []
        self.memory_counter = 0
        self.memory_length = 0
        self.env = env
        self.env_name = env_name
        self.her = HER_Energy_sampler(k_future)
    def __len__(self):
        return len(self.memory)
    def add(self, transition):
        if self.env_name in ['FetchPickAndPlace-v1','FetchPush-v1','FetchSlide-v1']:
            ag = np.stack(transition['achieved_goal'], axis=0)
            z_object_before = ag[:,2][0:-1]
            z_object_after = ag[:,2][1::]
            delta_z_object = z_object_after-z_object_before
            energy_potential = G*m*delta_z_object

            diff = np.diff(ag,axis=0)
            velocity = diff/Delta_t
            energy_kinetic = np.sum(0.5*m*np.power(velocity,2),axis=1)

            energy_total = weight_potential*energy_potential+weight_kinetic*energy_kinetic
            energy_total_diff = np.diff(energy_total)
            energy_transition = energy_total.copy()
            energy_transition[1::] = energy_total_diff.copy()
            energy_transition = np.clip(energy_transition,0,max_energy)
            energy_trajectory = np.sum(energy_transition)

            transition['energy_trajectory'] = energy_trajectory
            self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def sample_for_normalization(self, batchs):
        size = len(batchs[0]['next_state'])
        states, desired_goals = self.her.sample_for_normalization(batchs, size)
        return self.clip_obs(states), self.clip_obs(desired_goals)

    def sample(self, batch_size):

        states, actions, rewards, next_states, desired_goals = self.her.sample(self.memory, batch_size,
                                                                               self.env.compute_reward)
        return self.clip_obs(states), actions, rewards, self.clip_obs(next_states), self.clip_obs(desired_goals)

    @staticmethod
    def clip_obs(x):
        return np.clip(x, -200, 200)