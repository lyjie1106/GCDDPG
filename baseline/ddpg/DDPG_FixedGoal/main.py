import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from agent import Agent

ENV_NAME = 'Pendulum-v1'
MAX_EPISODES = 100
MAX_STEPS_PER_EP = 800

if __name__ == '__main__':

    env = gym.make(ENV_NAME)

    # define state space, action space, action boundary
    n_state = env.observation_space.shape
    n_action = env.action_space.shape[0]
    bound_action = [env.action_space.low[0], env.action_space.high[0]]

    # record loss
    global_episode_reward = []
    global_episode_actor_loss = []
    global_episode_critic_loss = []

    agent = Agent(n_state=n_state, n_action=n_action, bound_action=bound_action)

    pbar = tqdm(total=MAX_EPISODES)

    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        for step in range(MAX_STEPS_PER_EP):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state=state, reward=reward, done=done, action=action, next_state=next_state)

            if agent.check_if_learn():
                episode_reward += reward
                actor_loss, critic_loss = agent.learn()
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
            if done:
                break
            state = next_state
        global_episode_reward.append(
            episode_reward if episode == 0 else (global_episode_reward[-1] * 0.99 + 0.01 * episode_reward))
        global_episode_actor_loss.append(episode_actor_loss)
        global_episode_critic_loss.append(episode_critic_loss)

        pbar.update(1)
    pbar.close()
    # global_episode_reward.cpu()
    plt.figure()
    plt.subplot(311)
    plt.plot(np.arange(0, MAX_EPISODES), global_episode_reward)
    plt.title('reward')

    global_episode_actor_loss = torch.tensor(global_episode_actor_loss, device='cpu')
    plt.subplot(312)
    plt.plot(np.arange(0, MAX_EPISODES), global_episode_actor_loss)
    plt.title('actor_loss')

    global_episode_critic_loss = torch.tensor(global_episode_critic_loss, device='cpu')
    plt.subplot(313)
    plt.plot(np.arange(0, MAX_EPISODES), global_episode_critic_loss)
    plt.title('critic_loss')

    plt.show()
