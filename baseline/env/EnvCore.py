import gym
from gym_simple_minigrid.minigrid import SimpleMiniGridEnv
from gym import spaces

from gym.envs.registration import register

import numpy as np


class ModifiedMiniGridEnv(SimpleMiniGridEnv):
    def __init__(self, grid_size=None, width=None, height=None, max_steps=None, seed=9):
        # Env name
        self.name = self.__class__.__name__

        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size

        # Default max_steps
        if max_steps is None:
            max_steps = 4 * (width + height)

        # Action enumeration for this environment
        self.actions = SimpleMiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Observations are encoded as (x-coor, y-coor, orientation)
        self.observation_space = spaces.Dict({
            'achieved_goal': spaces.Box(
                low=np.array((0, 0)),
                high=np.array((width - 1, height - 1)),
                dtype=np.int
            ),
            'desired_goal': spaces.Box(
                low=np.array((0, 0)),
                high=np.array((width - 1, height - 1)),
                dtype=np.int
            ),
            'observation': spaces.Box(
                low=0,
                high=255,
                shape=(2+4,),
                dtype=np.int
            )
        })

        # Range of possible rewards
        self.reward_range = (-1, 0)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Initialize the RNG
        self.np_random = None
        self.seed(seed=seed)

        # Initialize the environment
        self.agent_pos = self.agent_dir = self.goal_pos = self.step_count = self.grid = self.goals = None
        self.reset()
    '''
    def reset_state_goal(self):
        agent_dir = self.np_random.randint(4)
        agent_x=agent_y = 0
        goal_x=goal_y = 4
        self.agent_pos=np.array((agent_x,agent_y))
        self.agent_dir = agent_dir
        self.goal_pos = np.array((goal_x,goal_y))
        return
    '''