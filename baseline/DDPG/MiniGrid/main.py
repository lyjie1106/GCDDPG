import os,sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append('/home/lujie/GCRL/PycharmProjects/GCRL-baseline')

import torch
import matplotlib.pyplot as plt
from agent import Agent
import numpy as np
from gym import envs
from copy import deepcopy
from mpi4py import MPI
import random



ENV_NAME = 'ModifiedFourRoomEnv-v0'
MAX_EPOCHS = 100
MAX_CYCLES = 50
MAX_EPISODES = 2
NUM_TRAIN = 40
MODEL_NAME='ModifiedFourRoomEnv-100e'

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['IN_MPI'] = '1'



def eval(env,agent,final=False):
    total_success_rate = []
    total_reward = []
    running_r = []
    success_time=0
    for ep in range(20):
        final_ep = (ep==19)
        ep_success_rate = []
        # reset ENV
        achieved_goal, desired_goal = 0, 0
        while np.array_equal(achieved_goal, desired_goal):
            env_dict = env.reset()
            state = env_dict['observation']
            achieved_goal = env_dict['achieved_goal']
            desired_goal = env_dict['desired_goal']
        ep_reward = 0
        for t in range(50):
            with torch.no_grad():
                actions,action_dicrete = agent.choose_action(state,desired_goal,ep,train_mode=False)
            next_env_dict, reward, done, info = env.step(env.actions(action_dicrete))
            '''
            if final_ep and final:
                print(next_env_dict)
            '''
            state = next_env_dict['observation'].copy()
            desired_goal = next_env_dict['desired_goal'].copy()
            ep_success_rate.append(info['is_success'])
            ep_reward+=reward
            if info['is_success']==1:
                success_time+=1
                break
        #total_success_rate.append(ep_success_rate)
        #total_reward.append(ep_reward if ep==0 else (total_reward[-1]*0.99+0.01*ep_reward))
    #total_success_rate = np.array(total_success_rate)
    #local_success_rate = np.mean(total_success_rate[:,-1])
    local_success_rate = success_time/20
    total_reward=0
    ep_reward=0
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate,op=MPI.SUM)
    return global_success_rate/MPI.COMM_WORLD.Get_size(),total_reward,ep_reward


def launch():
    # initial envcritic_loss
    env = envs.make(ENV_NAME)
    env.seed(MPI.COMM_WORLD.Get_rank())
    random.seed(MPI.COMM_WORLD.Get_rank())
    np.random.seed(MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(MPI.COMM_WORLD.Get_rank())

    n_state = env.observation_space.spaces['observation'].shape
    n_action = 1
    actions_num = 3
    n_goal = env.observation_space.spaces['desired_goal'].shape[0]
    #bound_action = [0,1,2]

    agent = Agent(n_state, n_action, n_goal, actions_num, deepcopy(env))
    # record loss in each epoch
    global_actor_loss = []
    global_critic_loss = []
    # record success rate
    global_success_rate = []

    for epoch in range(MAX_EPOCHS):
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        start_time = time.time()
        for cycle in range(MAX_CYCLES):
            mb = []
            cycle_actor_loss = 0
            cycle_critic_loss = 0
            for episode in range(MAX_EPISODES):
                episode_dict = {
                    'state': [],
                    'action': [],
                    'info': [],
                    'achieved_goal': [],
                    'desired_goal': [],
                    'next_state': [],
                    'next_achieved_goal': []
                }
                # reset ENV
                achieved_goal, desired_goal = 0, 0
                while np.array_equal(achieved_goal,desired_goal):
                    env_dict = env.reset()
                    state = env_dict['observation']
                    achieved_goal = env_dict['achieved_goal']
                    desired_goal = env_dict['desired_goal']
                # take action
                for t in range(50):
                    actions,action_dicrete = agent.choose_action(state, desired_goal,epoch)

                    next_env_dict, reward, done, info = env.step(action_dicrete)

                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]
                    next_desired_goal = next_env_dict["desired_goal"]

                    episode_dict['state'].append(state.copy())
                    episode_dict['action'].append(action_dicrete)
                    # episode_dict['info'].append(info.copy())
                    episode_dict['achieved_goal'].append(achieved_goal.copy())
                    episode_dict['desired_goal'].append(desired_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()
                episode_dict['state'].append(state.copy())
                episode_dict['achieved_goal'].append(achieved_goal.copy())
                episode_dict['desired_goal'].append(desired_goal.copy())
                episode_dict['next_state'] = episode_dict['state'][1:]
                episode_dict['next_achieved_goal'] = episode_dict['achieved_goal'][1:]
                mb.append(deepcopy(episode_dict))
            agent.store(mb)
            for n in range(NUM_TRAIN):
                actor_loss, critic_loss = agent.train(position=n,cycle = cycle)
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss
            epoch_actor_loss += cycle_actor_loss / NUM_TRAIN
            epoch_critic_loss += cycle_critic_loss / NUM_TRAIN
            agent.update_network()
        global_actor_loss.append(epoch_actor_loss / MAX_CYCLES)
        global_critic_loss.append(epoch_critic_loss / MAX_CYCLES)
        final = (epoch==(MAX_EPOCHS-1))
        success_rate,running_reward,episode_reward = eval(env,agent,final)
        # eval, and print train detail
        if MPI.COMM_WORLD.Get_rank() == 0:
            global_success_rate.append(success_rate)
            print('Epoch:%d|Running_time:%1f|Actor_loss:%3f|Critic_loss:%3f|success;:%3f'%(
                epoch,(time.time()-start_time),actor_loss,critic_loss,success_rate
            ))
    # plot train info after train
    if MPI.COMM_WORLD.Get_rank() == 0:
        agent.save_model(MODEL_NAME)

        plt.figure()
        plt.subplot(311)
        plt.plot(np.arange(0, MAX_EPOCHS), global_actor_loss)
        plt.title('actor loss')

        plt.subplot(312)
        plt.plot(np.arange(0, MAX_EPOCHS), global_critic_loss)
        plt.title('critic loss')

        plt.subplot(313)
        plt.plot(np.arange(0, MAX_EPOCHS), global_success_rate)
        plt.title('success rate')
        plt.savefig('./train_log.jpg')
        #plt.show()

        print('done')

if __name__ == '__main__':
    launch()
