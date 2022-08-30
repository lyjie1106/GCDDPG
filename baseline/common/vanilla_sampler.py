from copy import deepcopy

import numpy as np


class Vanilla_sampler:
    def __init__(self, config):
        self.Sampler_type = config['Sampler']

    def sample_for_normalization(self, batches, size):
        ep_indices = np.random.randint(0, len(batches), size)
        time_indices = np.random.randint(0, len(batches[0]['next_state']), size)

        states = []
        desired_goals = []
        for episode, timestep in zip(ep_indices, time_indices):
            states.append(deepcopy(batches[episode]['state'][timestep]))
            desired_goals.append((deepcopy(batches[episode]['desired_goal'][timestep])))
        states = np.vstack(states)
        desired_goals = np.vstack(desired_goals)

        return states, desired_goals

    def sample(self, memory, batch_size, compute_reward_func):
        ep_indices = np.random.randint(0, len(memory), batch_size)
        time_indices = np.random.randint(0, len(memory[0]['next_state']), batch_size)
        states = []
        actions = []
        desired_goals = []
        next_states = []
        next_achieved_goal = []
        for episode, timestep in zip(ep_indices, time_indices):
            states.append(memory[episode]['state'][timestep])
            actions.append(memory[episode]['action'][timestep])
            desired_goals.append(memory[episode]['desired_goal'][timestep])
            next_states.append(memory[episode]['next_state'][timestep])
            next_achieved_goal.append(memory[episode]['next_achieved_goal'][timestep])
        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_states = np.vstack(next_states)
        next_achieved_goals = np.vstack(next_achieved_goal)

        rewards = np.expand_dims(compute_reward_func(next_achieved_goals, desired_goals, None), 1)

        return states, actions, rewards, next_states, desired_goals
