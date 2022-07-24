import numpy as np
from copy import deepcopy
from her import HER_sampler
class Memory:
    def __init__(self,capacity,k_future,env):
        self.capacity = capacity
        self.memory = []
        self.memory_counter = 0
        self.memory_length = 0
        self.env = env
        #self.future_p = 1 - (1. / (1+k_future))
        self.her = HER_sampler(k_future)
    def __len__(self):
        return len(self.memory)

    def add(self,transition):
        self.memory.append(transition)
        if len(self.memory)>self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity
    @staticmethod
    def clip_obs(x):
        return np.clip(x,-200,200)
    def sample_for_normalization(self,batchs):
        size = len(batchs[0]['next_state'])
        '''
        # select which episode and which timesteps to be used
        ep_indices = np.random.randint(0,len(batchs),size)
        time_indices = np.random.randint(0,len(batchs[0]['next_state']),size)
        states = []
        desired_goals = []
        for episode,timestep in zip(ep_indices,time_indices):
            states.append(deepcopy(batchs[episode]['state'][timestep]))
            desired_goals.append((deepcopy(batchs[episode]['desired_goal'][timestep])))
        states = np.vstack(states)
        desired_goals = np.vstack(desired_goals)

        # for each transition, get a future offset from current goal to a new goal
        future_offset = np.random.uniform(size=size) * (len(batchs[0]['next_state']) - time_indices)
        future_offset = future_offset.astype(int)

        # HER
        # get indices of transition that use HER
        her_indices = np.where(np.random.uniform(size=size)<self.future_p)
        # get the index of new goal in the transition use HER
        future_t = (time_indices+1+future_offset)[her_indices]
        # replace goal with new goal
        future_ag = []
        for episode,f_offset in zip(ep_indices[her_indices],future_t):
            future_ag.append(deepcopy(batchs[episode]['achieved_goal'][f_offset]))
        future_ag = np.vstack(future_ag)
        desired_goals[her_indices] = future_ag

        return self.clip_obs(states),self.clip_obs(desired_goals)
        '''
        states, desired_goals = self.her.sample_for_normalization(batchs, size)
        return self.clip_obs(states), self.clip_obs(desired_goals)

    def sample(self,batch_size):
        '''

        # select which episode and which timesteps to be used
        ep_indices = np.random.randint(0,len(self.memory),batch_size)
        time_indices = np.random.randint(0,len(self.memory[0]['next_state']),batch_size)
        states = []
        actions = []
        desired_goals = []
        next_states = []
        next_achieved_goal = []
        for episode,timestep in zip(ep_indices,time_indices):
            states.append(self.memory[episode]['state'][timestep])
            actions.append(self.memory[episode]['action'][timestep])
            desired_goals.append(self.memory[episode]['desired_goal'][timestep])
            next_states.append(self.memory[episode]['next_state'][timestep])
            next_achieved_goal.append(self.memory[episode]['next_achieved_goal'][timestep])
        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_states = np.vstack(next_states)
        next_achieved_goals = np.vstack(next_achieved_goal)

        # for each transition, get a future offset from current goal to a new goal
        future_offset = np.random.uniform(size=batch_size) * (len(self.memory[0]['next_state']) - time_indices)
        future_offset = future_offset.astype(int)

        #HER
        # get indices of transition that use HER
        her_indices = np.where(np.random.uniform(size=batch_size)<self.future_p)
        # get the index of new goal in the transition use HER
        future_t = (time_indices + 1 + future_offset)[her_indices]
        # replace goal with new goal
        future_ag = []
        for episode,f_offset in zip(ep_indices[her_indices],future_t):
            future_ag.append(deepcopy(self.memory[episode]['achieved_goal'][f_offset]))
        future_ag = np.vstack(future_ag)
        desired_goals[her_indices] = future_ag

        rewards = np.expand_dims(self.env.compute_reward(next_achieved_goals,desired_goals,None),1)
        return self.clip_obs(states),actions,rewards,self.clip_obs(next_states),self.clip_obs(desired_goals)
        '''
        states, actions, rewards, next_states, desired_goals = self.her.sample(self.memory, batch_size,self.env.compute_reward)
        return self.clip_obs(states), actions, rewards, self.clip_obs(next_states), self.clip_obs(desired_goals)
