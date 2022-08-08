import numpy as np
from baseline.common.her_ebp import HER_Energy_sampler


class MemoryEnergy:
    def __init__(self, env, config):
        self.memory = []
        self.memory_counter = 0
        self.memory_length = 0
        self.env = env
        self.capacity = config['MEMORY_CAPACITY']
        self.G = config['G']
        self.m = config['M']
        self.delta_t = config['Delta_t']
        self.weight_potential = config['Weight_potential']
        self.weight_kinetic = config['Weight_kinetic']
        self.max_energy = config['Max_energy']
        self.sampler = HER_Energy_sampler(config)

    def __len__(self):
        return len(self.memory)

    def add(self, transition):
        ag = np.stack(transition['achieved_goal'], axis=0)
        z_object_before = ag[:, 2][0:-1]
        z_object_after = ag[:, 2][1::]
        delta_z_object = z_object_after - z_object_before
        energy_potential = self.G * self.m * delta_z_object

        diff = np.diff(ag, axis=0)
        velocity = diff / self.delta_t
        energy_kinetic = np.sum(0.5 * self.m * np.power(velocity, 2), axis=1)

        energy_total = self.weight_potential * energy_potential + self.weight_kinetic * energy_kinetic
        energy_total_diff = np.diff(energy_total)
        energy_transition = energy_total.copy()
        energy_transition[1::] = energy_total_diff.copy()
        energy_transition = np.clip(energy_transition, 0, self.max_energy)
        energy_trajectory = np.sum(energy_transition)

        transition['energy_trajectory'] = energy_trajectory
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def sample_for_normalization(self, batches):
        size = len(batches[0]['next_state'])
        states, desired_goals = self.sampler.sample_for_normalization(batches, size)
        return self.clip_obs(states), self.clip_obs(desired_goals)

    def sample(self, batch_size):
        states, actions, rewards, next_states, desired_goals = self.sampler.sample(self.memory, batch_size,
                                                                                   self.env.compute_reward)
        return self.clip_obs(states), actions, rewards, self.clip_obs(next_states), self.clip_obs(desired_goals)

    @staticmethod
    def clip_obs(x):
        return np.clip(x, -200, 200)
