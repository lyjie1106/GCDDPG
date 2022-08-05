from copy import deepcopy

import numpy as np
from gym import envs
from torch import device
import torch
from baseline.DDPG.Mujoco.agent import Agent as Agent_Mujoco
from baseline.DDPG.MiniGrid.agent import Agent as Agent_Minigrid

ENV_TYPE = 'mujoco'
ENV_NAME = 'FetchSlide-v1'
MODEL_NAME = '../../data/2022-08-05 04:57:23-FetchSlide-v1/50.pth'
MAX_EPISODES = 100

env = envs.make(ENV_NAME)
env.seed(11)

if ENV_TYPE == 'minigrid':
    n_state = env.observation_space.spaces['observation'].shape
    n_action = 1
    actions_num = 3
    n_goal = env.observation_space.spaces['desired_goal'].shape[0]
    agent = Agent_Minigrid(n_state, n_action, n_goal, actions_num, deepcopy(env))
elif ENV_TYPE == 'mujoco':
    n_state = env.observation_space.spaces['observation'].shape
    n_action = env.action_space.shape[0]
    n_goal = env.observation_space.spaces['desired_goal'].shape[0]
    bound_action = [env.action_space.low[0], env.action_space.high[0]]
    agent = Agent_Mujoco(n_state, n_action, n_goal, bound_action, deepcopy(env))
else:
    print('wrong env type')

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
    done = False
    for t in range(50):
        if ENV_TYPE == 'minigrid':
            actions, action_dicrete = agent.choose_action(state, desired_goal, i, train_mode=False)
            next_env_dict, reward, done, info = env.step(env.actions(action_dicrete))
            state = next_env_dict['observation'].copy()
            desired_goal = next_env_dict['desired_goal'].copy()
        else:
            actions = agent.choose_action(state, desired_goal, train_mode=False)
            next_env_dict, reward, done, info = env.step(actions)
            state = next_env_dict['observation'].copy()
            desired_goal = next_env_dict['desired_goal'].copy()
        '''
        plt.imshow(env.render())
        plt.pause(0.1)
        plt.ioff()
        '''
        if info['is_success'] == 1:
            success += 1
            break
print(success / MAX_EPISODES)
