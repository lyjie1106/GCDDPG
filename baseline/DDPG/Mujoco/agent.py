import numpy as np

from baseline.DDPG.Mujoco.models import Actor, Critic
from torch import from_numpy, device
import torch
from mpi4py import MPI
from torch.optim import Adam

from baseline.common.memory import Memory
from baseline.common.memory_ebp import MemoryEnergy
from baseline.common.normalizer import Normalizer

ACTOR_LOSS_L2 = 1  # L2 regularization


class Agent:
    def __init__(self, n_state, n_action, n_goal, bound_action, env, config):
        self.n_state = n_state
        self.n_action = n_action
        self.n_goal = n_goal
        self.bound_action = bound_action
        self.env = env
        self.env_name = config['Env_name']
        self.MEMORY_CAPACITY = config['MEMORY_CAPACITY']
        self.K_future = config['K_future']
        self.LR_A = config['LR_A']
        self.LR_C = config['LR_C']
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.GAMMA = config['GAMMA']
        self.TAU = config['TAU']
        self.Sampler = config['Sampler']

        self.device = device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = Actor(self.n_state, self.n_action, self.n_goal).to(self.device)
        self.critic = Critic(self.n_state, self.n_action, self.n_goal).to(self.device)
        self.sync_network(self.actor)
        self.sync_network(self.critic)
        self.actor_target = Actor(self.n_state, self.n_action, self.n_goal).to(self.device)
        self.critic_target = Critic(self.n_state, self.n_action, self.n_goal).to(self.device)
        self.hard_update_network(self.actor, self.actor_target)
        self.hard_update_network(self.critic, self.critic_target)

        if self.Sampler == 'EBPHER':
            self.memory = MemoryEnergy(self.env, config)
        else:
            self.memory = Memory(self.env, config)

        self.actor_optimizer = Adam(self.actor.parameters(), self.LR_A)
        self.critic_optimizer = Adam(self.critic.parameters(), self.LR_C)

        self.state_normalizer = Normalizer(self.n_state[0], default_clip_range=5)
        self.goal_normalizer = Normalizer(self.n_goal, default_clip_range=5)

    # choose action
    def choose_action(self, state, goal, train_mode=True):
        state = self.state_normalizer.normalize(state)
        goal = self.goal_normalizer.normalize(goal)
        state = np.expand_dims(state, axis=0)
        goal = np.expand_dims(goal, axis=0)

        with torch.no_grad():
            x = np.concatenate([state, goal], axis=1)
            x = from_numpy(x).float().to(self.device)
            action = self.actor(x)[0].cpu().data.numpy()
        if train_mode:
            # add noise
            action += 0.2 * np.random.randn(self.n_action)
            action = np.clip(action, self.bound_action[0], self.bound_action[1])
            random_action = np.random.uniform(low=self.bound_action[0], high=self.bound_action[1], size=self.n_action)
            action += np.random.binomial(1, 0.3, 1)[0] * (random_action - action)
        return action

    # store minibatch into memory
    def store(self, minibatch):
        for batch in minibatch:
            self.memory.add(batch)
        self._update_normalizer(minibatch)

    # train
    def train(self):
        states, actions, rewards, next_states, goals = self.memory.sample(self.BATCH_SIZE)

        states = self.state_normalizer.normalize(states)
        next_states = self.state_normalizer.normalize(next_states)
        goals = self.goal_normalizer.normalize(goals)

        current_states_goals = np.concatenate([states, goals], axis=1)
        next_states_goals = np.concatenate([next_states, goals], axis=1)

        current_states_goals = torch.Tensor(current_states_goals).to(self.device)
        next_states_goals = torch.Tensor(next_states_goals).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        actions = torch.Tensor(actions).to(self.device)

        # calculate critic loss
        with torch.no_grad():
            target_action = self.actor_target(next_states_goals)
            target_q = self.critic_target(next_states_goals, target_action)
            # target_q = target_q.detach()
            # target_return = rewards + GAMMA * target_q.detach()*(-rewards)
            target_return = rewards + self.GAMMA * target_q.detach()
            # target_return = target_return.detach()
            # clip the return, due to the reward in env is non-positive
            clip_return = 1 / (1 - self.GAMMA)
            target_return = torch.clamp(target_return, -clip_return, 0)
        q_eval = self.critic(current_states_goals, actions)
        critic_loss = (target_return - q_eval).pow(2).mean()

        # calculate actor loss
        new_actions = self.actor(current_states_goals)
        actor_loss = -self.critic(current_states_goals, new_actions).mean()
        # l2 Regularization: avoid move too much
        actor_loss += ACTOR_LOSS_L2 * (new_actions.pow(2) / self.bound_action[1]).mean()

        # optimize actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.sync_grads(self.actor)
        self.actor_optimizer.step()

        # optimize critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.sync_grads(self.critic)
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    # hard update network parameters
    @staticmethod
    def hard_update_network(local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())

    # soft update target network(ExponentialMovingAverage)
    @staticmethod
    def soft_update_network(local_model, target_model, TAU):
        for target_params, local_params in zip(target_model.parameters(), local_model.parameters()):
            target_params.data.copy_(TAU * local_params.data + (1 - TAU) * target_params.data)

    # sync network across all process
    @staticmethod
    def sync_network(network):
        comm = MPI.COMM_WORLD
        flat_params = _get_flat_params_or_grads(network, mode='params')
        comm.Bcast(flat_params, root=0)
        _set_flat_params_or_grads(network, flat_params, mode='params')

    # sync gradient across all process
    @staticmethod
    def sync_grads(network):
        flat_grads = _get_flat_params_or_grads(network, mode='grads')
        comm = MPI.COMM_WORLD
        global_grad = np.zeros_like(flat_grads)
        # according to https://github.com/TianhongDai/hindsight-experience-replay/issues/4, sum works better than average
        comm.Allreduce(flat_grads, global_grad, op=MPI.SUM)
        _set_flat_params_or_grads(network, global_grad, mode='grads')

    # update normalizer
    def _update_normalizer(self, minibatch):
        states, goals = self.memory.sample_for_normalization(minibatch)
        self.state_normalizer.update(states)
        self.goal_normalizer.update(goals)
        self.state_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()

    # update network
    def update_network(self):
        self.soft_update_network(self.actor, self.actor_target, self.TAU)
        self.soft_update_network(self.critic, self.critic_target, self.TAU)

    def save_model(self, name):
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'state_normalizer_mean': self.state_normalizer.mean,
                    'state_normalizer_std': self.state_normalizer.std,
                    'goal_normalizer_mean': self.goal_normalizer.mean,
                    'goal_normalizer_std': self.goal_normalizer.std},
                   name)

    def load_model(self, name):
        checkpoint = torch.load(name)
        actor_state_dict = checkpoint['actor_state_dict']
        self.actor.load_state_dict(actor_state_dict)
        state_normalizer_mean = checkpoint['state_normalizer_mean']
        self.state_normalizer.mean = state_normalizer_mean
        state_normalizer_std = checkpoint['state_normalizer_std']
        self.state_normalizer.std = state_normalizer_std
        goal_normalizer_mean = checkpoint['goal_normalizer_mean']
        self.goal_normalizer.mean = goal_normalizer_mean
        goal_normalizer_std = checkpoint['goal_normalizer_std']
        self.goal_normalizer.std = goal_normalizer_std

    def set_eval_mode(self):
        self.actor.eval()


# get flat parameters or grads of network
def _get_flat_params_or_grads(network, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


# set flat parameters or grads of network
def _set_flat_params_or_grads(network, flat_params, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()
