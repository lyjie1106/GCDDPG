import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from gym import envs
from copy import deepcopy
from mpi4py import MPI
import random
from baseline.DDPG.Mujoco.agent import Agent as Agent_Mujoco
from baseline.DDPG.MiniGrid.agent import Agent as Agent_Minigrid
import json

data_dir_path = './data'


def train(env, agent, config):
    env_name = config['Env_name']
    max_epochs = config['MAX_EPOCHS']
    max_cycles = config['MAX_CYCLES']
    max_episodes = config['MAX_EPISODES']
    num_train = config['NUM_TRAIN']
    env_type = config['Env_type']

    dir_name = ''
    if MPI.COMM_WORLD.Get_rank() == 0:
        train_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        dir_name = '{}-{}'.format(train_time, env_name)
        mkdir('{}/{}'.format(data_dir_path, dir_name))
        save_config(config, '{}/{}/config.json'.format(data_dir_path, dir_name))
    dir_name = MPI.COMM_WORLD.bcast(dir_name, root=0)

    # record loss in each epoch
    global_actor_loss = []
    global_critic_loss = []
    # record success rate
    global_success_rate = []

    for epoch in range(max_epochs):
        epoch_actor_loss, epoch_critic_loss = 0, 0
        start_time = time.time()
        for cycle in range(max_cycles):
            minibatch = []
            cycle_actor_loss, cycle_critic_loss = 0, 0
            for episode in range(max_episodes):
                episode_dict = {
                    'state': [],
                    'action': [],
                    'info': [],
                    'achieved_goal': [],
                    'desired_goal': [],
                    'next_state': [],
                    'next_achieved_goal': []
                }
                achieved_goal, desired_goal = 0, 0  # reset ENV
                while check_if_achieved(achieved_goal, desired_goal, env_type):
                    env_dict = env.reset()
                    state = env_dict['observation']
                    achieved_goal = env_dict['achieved_goal']
                    desired_goal = env_dict['desired_goal']
                # take action
                for t in range(50):
                    if env_type == 'minigrid':
                        action_continuous, action = agent.choose_action(state, desired_goal, epoch)
                        next_env_dict, reward, done, info = env.step(action)
                    else:
                        action = agent.choose_action(state, desired_goal)
                        next_env_dict, reward, done, info = env.step(action)
                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]
                    next_desired_goal = next_env_dict["desired_goal"]
                    # record transition
                    episode_dict['state'].append(state.copy())
                    episode_dict['action'].append(action if env_type == 'minigrid' else action.copy())
                    episode_dict['achieved_goal'].append(achieved_goal.copy())
                    episode_dict['desired_goal'].append(desired_goal.copy())
                    # update state
                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()
                episode_dict['state'].append(state.copy())
                episode_dict['achieved_goal'].append(achieved_goal.copy())
                episode_dict['desired_goal'].append(desired_goal.copy())
                episode_dict['next_state'] = episode_dict['state'][1:]
                episode_dict['next_achieved_goal'] = episode_dict['achieved_goal'][1:]
                minibatch.append(deepcopy(episode_dict))
            agent.store(minibatch)
            for n in range(num_train):
                actor_loss, critic_loss = agent.train()
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss
            epoch_actor_loss += cycle_actor_loss / num_train
            epoch_critic_loss += cycle_critic_loss / num_train
            agent.update_network()
        success_rate = eval(env, agent, env_type)
        time_duration = time.time() - start_time
        global_success_rate.append(success_rate)
        global_actor_loss.append(epoch_actor_loss / max_cycles)
        global_critic_loss.append(epoch_critic_loss / max_cycles)
        print_train_info(epoch, time_duration, actor_loss, critic_loss, success_rate)
        save_model(epoch + 1, agent, dir_name)
    if MPI.COMM_WORLD.Get_rank() == 0:
        plt.figure()
        plt.subplot(311)
        plt.plot(np.arange(0, max_epochs), global_actor_loss)
        plt.title('actor loss')

        plt.subplot(312)
        plt.plot(np.arange(0, max_epochs), global_critic_loss)
        plt.title('critic loss')

        plt.subplot(313)
        plt.plot(np.arange(0, max_epochs), global_success_rate)
        plt.title('success rate')
        plt.savefig('{}/{}/train_log.jpg'.format(data_dir_path, dir_name))


def save_config(config, path):
    json.dump(config, open(path, 'w'))


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.mkdir(path)


def save_model(epoch, agent, dir_name):
    if MPI.COMM_WORLD.Get_rank() == 0 and epoch % 10 == 0:
        agent.save_model('{}/{}/{}.pth'.format(data_dir_path, dir_name, epoch))


def print_train_info(epoch, time_duration, actor_loss, critic_loss, success_rate):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print('Epoch:%d|Running_time:%1f|Actor_loss:%3f|Critic_loss:%3f|success;:%3f' % (
            epoch, time_duration, actor_loss, critic_loss, success_rate
        ))


def check_if_achieved(achieved_goal, desired_goal, env_type):
    if env_type == 'minigrid':
        result = np.array_equal(achieved_goal, desired_goal)
    else:
        result = np.linalg.norm(achieved_goal - desired_goal) <= 0.05
    return result


def eval(env, agent, env_type):
    success_time = 0
    for ep in range(20):
        ep_success_rate = []
        achieved_goal, desired_goal = 0, 0
        while check_if_achieved(achieved_goal, desired_goal, env_type):
            env_dict = env.reset()
            state = env_dict['observation']
            achieved_goal = env_dict['achieved_goal']
            desired_goal = env_dict['desired_goal']
        for t in range(50):
            if env_type == 'minigrid':
                action_continuous, action = agent.choose_action(state, desired_goal, ep, train_mode=False)
                next_env_dict, reward, done, info = env.step(action)
            else:
                action = agent.choose_action(state, desired_goal, train_mode=False)
                next_env_dict, reward, done, info = env.step(action)
            state = next_env_dict['observation'].copy()
            desired_goal = next_env_dict['desired_goal'].copy()
            ep_success_rate.append(info['is_success'])
            if info['is_success'] == 1:
                success_time += 1
                break
    local_success_rate = success_time / 20
    global_success_rate = (MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)) / (MPI.COMM_WORLD.Get_size())
    return global_success_rate


def launch(config):
    env_type = config['Env_type']
    env_name = config['Env_name']

    env = envs.make(env_name)
    env.seed(MPI.COMM_WORLD.Get_rank())
    random.seed(MPI.COMM_WORLD.Get_rank())
    np.random.seed(MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(MPI.COMM_WORLD.Get_rank())

    if env_type == 'minigrid':
        n_state = env.observation_space.spaces['observation'].shape
        n_action = 1
        actions_num = 3
        n_goal = env.observation_space.spaces['desired_goal'].shape[0]
        agent = Agent_Minigrid(n_state, n_action, n_goal, actions_num, deepcopy(env), config)
    elif env_type == 'mujoco':
        n_state = env.observation_space.spaces['observation'].shape
        n_action = env.action_space.shape[0]
        n_goal = env.observation_space.spaces['desired_goal'].shape[0]
        bound_action = [env.action_space.low[0], env.action_space.high[0]]
        agent = Agent_Mujoco(n_state, n_action, n_goal, bound_action, deepcopy(env), config)
    else:
        print('wrong env type')

    train(env, agent, config)
