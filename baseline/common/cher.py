import random
from copy import deepcopy

import numpy as np
from sklearn.neighbors import NearestNeighbors


class CHER_sampler:
    def __init__(self, config):
        self.K_future = config['K_future']
        self.learning_rate = config['LR_CHER']
        self.learning_step = 0
        self.lamda0 = config['LAMDA_0']
        self.fixed_lamda = config['FIXED_LAMDA']
        self.size_A = config['SIZE_A']
        self.size_k = config['SIZE_k']
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
        ep_indices = np.random.randint(0, len(memory), batch_size)
        time_indices = np.random.randint(0, len(memory[0]['next_state']), batch_size)
        states = []
        actions = []
        achieved_goals = []
        desired_goals = []
        next_states = []
        next_achieved_goal = []
        for episode, timestep in zip(ep_indices, time_indices):
            states.append(memory[episode]['state'][timestep])
            actions.append(memory[episode]['action'][timestep])
            achieved_goals.append(memory[episode]['achieved_goal'][timestep])
            desired_goals.append(memory[episode]['desired_goal'][timestep])
            next_states.append(memory[episode]['next_state'][timestep])
            next_achieved_goal.append(memory[episode]['next_achieved_goal'][timestep])
        states = np.vstack(states)
        actions = np.vstack(actions)
        achieved_goals = np.vstack(achieved_goals)
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

        states_cher, actions_cher, rewards_cher, next_states_cher, achieved_goals_cher, desired_goals_cher = self.curriculum(
            states, actions, rewards, next_states, achieved_goals, desired_goals)

        return states_cher, actions_cher, rewards_cher, next_states_cher, desired_goals_cher

    # curriculum HER
    def curriculum(self, states, actions, rewards, next_states, achieved_goals, desired_goals):
        set_list = self.lazier_and_goals_sample(desired_goals, achieved_goals)
        states_cher = states[set_list]
        actions_cher = actions[set_list]
        rewards_cher = rewards[set_list]
        next_states_cher = next_states[set_list]
        achieved_goals_cher = achieved_goals[set_list]
        desired_goals_cher = desired_goals[set_list]

        self.learning_step += 1
        return states_cher, actions_cher, rewards_cher, next_states_cher, achieved_goals_cher, desired_goals_cher

    # Lazier than Lazy Greedy, k=3
    def lazier_and_goals_sample(self, desired_goals, achieved_goal):
        num_neighbor = 1
        # get k-neighbors_graph of desired_goals, kgraph[i][j]=d means goal j is the closest neighbor of goal i, with distance d
        kgraph = NearestNeighbors(n_neighbors=num_neighbor, algorithm='kd_tree', metric='euclidean').fit(
            desired_goals).kneighbors_graph(mode='distance').tocoo(copy=False)

        if np.sum(kgraph.data) == 0:
            kgraph.data[np.where(kgraph.data == 0)] = 1

        # set of row and col that each element in
        row = kgraph.row
        col = kgraph.col
        # similarity of goals
        sim = np.exp(-np.divide(np.power(kgraph.data, 2), np.mean(kgraph.data) ** 2))
        delta = np.mean(kgraph.data)
        # store all idx of selected tuples
        sel_idx_set = []
        # initial an idx set of transition tuples
        idx_set = [i for i in range(len(desired_goals))]
        # initial lamda
        balance = self.fixed_lamda
        if int(balance) == -1:
            balance = np.power(1 + self.learning_rate, self.learning_step) * self.lamda0
        v_set = [i for i in range(len(desired_goals))]
        max_set = []
        # size(A)=SIZE_A
        for i in range(0, self.size_A):
            # minibatch size = 3
            sub_size = 3
            # randomly select a minibatch
            sub_set = random.sample(idx_set, sub_size)
            # initial selected idx
            sel_idx = -1
            # initial max value=-inf
            max_marginal = float('-inf')
            # iterate minibatch A of size k
            for j in range(sub_size):
                k_idx = sub_set[j]
                # a_set store maximum(gl,gj) that gl belong to A, gj belong to B
                marginal_v, new_a_set = self.fa(k_idx, max_set, v_set, sim, row, col)
                distance = np.linalg.norm(desired_goals[sub_set[j]] - achieved_goal[sub_set[j]])
                # controversial: not follow Eq.6
                marginal_v = marginal_v - balance * distance
                # if it finds bigger value
                if marginal_v > max_marginal:
                    # update selected idx
                    sel_idx = k_idx
                    # update maximum value
                    max_marginal = marginal_v
                    max_set = new_a_set
            idx_set.remove(sel_idx)
            sel_idx_set.append(sel_idx)
        return np.array(sel_idx_set)

    # calculate F(i|A),and update A
    @staticmethod
    def fa(k, a_set, v_set, sim, row, col):
        # if A is empty
        if len(a_set) == 0:
            init_a_set = []
            marginal_v = 0
            for i in v_set:
                max_ki = 0
                if k == col[i]:
                    max_ki = sim[i]
                init_a_set.append(max_ki)
                marginal_v += max_ki
            return marginal_v, init_a_set
        new_a_set = []
        marginal_v = 0
        for i in v_set:
            sim_ik = 0
            if k == col[i]:
                sim_ik = sim[i]
            # if sim(gi,gj)-maximum(gl,gj)>0, F(i|A)+=sim(gi,gj)-maximum(gl,gj)
            if sim_ik > a_set[i]:
                max_ki = sim_ik
                new_a_set.append(max_ki)
                marginal_v += max_ki - a_set[i]
            # else F(i|A)+=0
            else:
                new_a_set.append(a_set[i])
        return marginal_v, new_a_set
