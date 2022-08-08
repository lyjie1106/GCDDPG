import numpy as np
import torch
from torch import from_numpy,device
from models import Actor,Critic
from memory import Memory,Transition
from torch.optim import Adam

GAMMA = 0.9
LR_A = 0.001
LR_C = 0.002
TAU = 0.01
VAR=1

MEMORY_CAPACITY = 1000
BATCH_SIZE=64


class Agent:
    def __init__(self,n_state,n_action,bound_action):
        self.device = device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_state = n_state
        self.n_action = n_action
        self.bound_action = bound_action

        self.actor = Actor(self.n_state).to(self.device)
        self.critic = Critic(self.n_state,self.n_action).to(self.device)
        self.actor_target = Actor(self.n_state).to(self.device)
        self.critic_target = Critic(self.n_state, self.n_action).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.eval()
        self.critic_target.eval()

        self.memory = Memory(MEMORY_CAPACITY)
        self.actor_optimizer = Adam(self.actor.parameters(),LR_A)
        self.critic_optimizer = Adam(self.critic.parameters(),LR_C,weight_decay=1e-2)

        self.critic_loss = torch.nn.MSELoss()

    def choose_action(self,state):
        state = np.expand_dims(state,axis=0)
        state = from_numpy(state).float().to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)[0].cpu().data.numpy()
        self.actor.train()
        action = np.clip(np.random.normal(action, VAR), self.bound_action[0], self.bound_action[1])
        return action
    def store_transition(self,state,reward,done,action,next_state):
        state = from_numpy(state).float().to('cpu')
        reward = torch.FloatTensor([reward])
        done = torch.Tensor([done])
        action = from_numpy(action)
        next_state = from_numpy(next_state).float().to('cpu')

        self.memory.add(state,reward,done,action,next_state)

    #ExponentialMovingAverage
    def ema_update(self,model,target_model):
        for i,j in zip(target_model.parameters(),model.parameters()):
            i.data.copy_(TAU*j.data+(1-TAU)*i.data)

    def learn(self):
        batch = self.memory.sample(BATCH_SIZE)
        states,actions,rewards,dones,next_states = Memory.unpack_batch(batch,BATCH_SIZE,self.n_state,self.n_action)

        with torch.no_grad():
            target_action = self.actor_target(next_states)
            target_q = self.critic_target(next_states,target_action)
            target_return = rewards+GAMMA*target_q*(1-dones)
        q_eval = self.critic(states,actions)
        critic_loss = self.critic_loss(target_return.view(BATCH_SIZE,1),q_eval)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states,self.actor(states))
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.ema_update(self.actor,self.actor_target)
        self.ema_update(self.critic,self.critic_target)

        return actor_loss,critic_loss

    def check_if_learn(self):
        return False if len(self.memory)<BATCH_SIZE else True