import gym
import matplotlib.pyplot as plt
from env.ModifiedFourRoomEnv import ModifiedFourRoomEnv

env = gym.make('ModifiedFourRoomEnv-v0')

obs = env.reset()
plt.imshow(env.render())

print('')