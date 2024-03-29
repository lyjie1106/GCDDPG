import json
import sys
from copy import deepcopy

import numpy as np
import torch
from gym import envs
from matplotlib import pyplot as plt
from torch import device
import argparse

from baseline.ddpg.minigrid.agent import Agent as Agent_Minigrid
from baseline.ddpg.mujoco.agent import Agent as Agent_Mujoco

def get_args():
    parser = argparse.ArgumentParser(prog='Deep Deterministic Policy Gradient-play demo',
                                     description='DDPG with different relabeling algorithm')
    parser.add_argument('--model', dest='model_folder_path', required=True, help='File path of model folder')
    args = parser.parse_args()
    return args

def read_config(path):
    with open('{}/config.json'.format(path), 'r') as f:
        config = json.load(f)
    return config


if __name__ == '__main__':
    args = get_args()
    folder_path = args.model_folder_path
    config = read_config(folder_path)
    Env_type = config['Env_type']
    Env_name = config['Env_name']
    MAX_EPISODES = 100

    env = envs.make(Env_name)

    if Env_type == 'minigrid':
        n_state = env.observation_space.spaces['observation'].shape
        n_action = 1
        actions_num = 3
        n_goal = env.observation_space.spaces['desired_goal'].shape[0]
        agent = Agent_Minigrid(n_state, n_action, n_goal, actions_num, deepcopy(env), config)
    elif Env_type == 'mujoco':
        n_state = env.observation_space.spaces['observation'].shape
        n_action = env.action_space.shape[0]
        n_goal = env.observation_space.spaces['desired_goal'].shape[0]
        bound_action = [env.action_space.low[0], env.action_space.high[0]]
        agent = Agent_Mujoco(n_state, n_action, n_goal, bound_action, deepcopy(env), config)
    else:
        print('wrong env type')

    agent.load_model('{}/50.pth'.format(folder_path))
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
            if Env_type == 'minigrid':
                actions, action_dicrete = agent.choose_action(state, desired_goal, i, train_mode=False)
                next_env_dict, reward, done, info = env.step(env.actions(action_dicrete))
                state = next_env_dict['observation'].copy()
                desired_goal = next_env_dict['desired_goal'].copy()
            else:
                actions = agent.choose_action(state, desired_goal, train_mode=False)
                next_env_dict, reward, done, info = env.step(actions)
                state = next_env_dict['observation'].copy()
                desired_goal = next_env_dict['desired_goal'].copy()

            if Env_type == 'minigrid':
                plt.imshow(env.render())
                plt.pause(0.1)
                plt.ioff()
            else:
                env.render()
            if info['is_success'] == 1:
                success += 1
                break
    print(success / MAX_EPISODES)
