import numpy as np
from copy import deepcopy


class HER_sampler:
    def __init__(self, config):
        self.K_future = config['K_future']
        self.future_p = 1 - (1. / (1 + self.K_future))

    def sample_for_normalization(self, batches, size):
        # select which episode and which timestep to be used
        ep_indices = np.random.randint(0, len(batches), size)
        time_indices = np.random.randint(0, len(batches[0]['next_state']), size)

        states = []
        desired_goals = []
        for episode, timestep in zip(ep_indices, time_indices):
            states.append(deepcopy(batches[episode]['state'][timestep]))
            desired_goals.append((deepcopy(batches[episode]['desired_goal'][timestep])))
        states = np.vstack(states)
        desired_goals = np.vstack(desired_goals)

        # for each transition, get a future offset from current goal to a new goal
        future_offset = np.random.uniform(size=size) * (len(batches[0]['next_state']) - time_indices)
        future_offset = future_offset.astype(int)

        # HER
        # get indices of transition that use HER
        her_indices = np.where(np.random.uniform(size=size) < self.future_p)
        # get the index of new goal in the transition use HER
        future_t = (time_indices + 1 + future_offset)[her_indices]
        # replace goal with new goal
        future_ag = []
        for episode, f_offset in zip(ep_indices[her_indices], future_t):
            future_ag.append(deepcopy(batches[episode]['achieved_goal'][f_offset]))
        future_ag = np.vstack(future_ag)
        desired_goals[her_indices] = future_ag

        return states, desired_goals

    def sample(self, memory, batch_size, compute_reward_func):
        # select which episode and which timestep to be used
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

        # for each transition, get a future offset from current goal to a new goal
        future_offset = np.random.uniform(size=batch_size) * (len(memory[0]['next_state']) - time_indices)
        future_offset = future_offset.astype(int)

        # HER
        # get indices of transition that use HER
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        # get the index of new goal in the transition use HER
        future_t = (time_indices + 1 + future_offset)[her_indices]
        # replace goal with new goal
        future_ag = []
        for episode, f_offset in zip(ep_indices[her_indices], future_t):
            future_ag.append(deepcopy(memory[episode]['achieved_goal'][f_offset]))
        future_ag = np.vstack(future_ag)
        desired_goals[her_indices] = future_ag

        rewards = np.expand_dims(compute_reward_func(next_achieved_goals, desired_goals, None), 1)

        return states, actions, rewards, next_states, desired_goals
