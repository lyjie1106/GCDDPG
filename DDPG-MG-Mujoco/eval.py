from copy import deepcopy

from agent import Agent
import numpy as np
import gym_simple_minigrid
import gym
from gym import envs
from torch import device
import torch
import matplotlib.pyplot as plt
from env.ModifiedFourRoomEnv import ModifiedFourRoomEnv

ENV_NAME = 'FetchSlide-v1'
MODEL_NAME='FetchSlide-50e-CHER'
MAX_EPISODES = 100

env = envs.make(ENV_NAME)
env.seed(9)

n_state = env.observation_space.spaces['observation'].shape
n_action = env.action_space.shape[0]
n_goal = env.observation_space.spaces['desired_goal'].shape[0]
bound_action = [env.action_space.low[0], env.action_space.high[0]]
agent = Agent(n_state, n_action, n_goal, bound_action, deepcopy(env))


agent.load_model(MODEL_NAME)
agent.set_eval_mode()
device = device('cuda' if torch.cuda.is_available() else 'cpu')

success = 0
for i in range(MAX_EPISODES):
    achieved_goal, desired_goal = 0, 0
    while np.array_equal(achieved_goal, desired_goal):
        env_dict = env.reset()
        state = env_dict['observation']
        achieved_goal = env_dict['achieved_goal']
        desired_goal = env_dict['desired_goal']
    done=False
    for t in range(50):
        actions= agent.choose_action(state, desired_goal, train_mode=False)
        next_env_dict, reward, done, info = env.step(actions)
        state = next_env_dict['observation'].copy()
        desired_goal = next_env_dict['desired_goal'].copy()
        '''
        plt.imshow(env.render())
        plt.pause(0.1)
        plt.ioff()
        '''
        if info['is_success'] == 1:
            success+=1
            break
print(success/MAX_EPISODES)