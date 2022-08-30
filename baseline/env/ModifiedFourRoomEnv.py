import numpy as np
from gym_simple_minigrid.minigrid import Wall

from baseline.env.EnvCore import ModifiedMiniGridEnv

# Map of agent direction indices to vectors
DIRS = [
    # Right (positive X)
    (1, 0),
    # Down (positive Y)
    (0, 1),
    # Left (negative X)
    (-1, 0),
    # Up (negative Y)
    (0, -1),
]
DIR_TO_VEC = {i: np.array(d) for i, d in enumerate(DIRS)}


class ModifiedFourRoomEnv(ModifiedMiniGridEnv):
    def __init__(self):
        super().__init__(grid_size=9)

    def reset(self):
        # Step count since episode start
        self.step_count = 0

        # Create grid
        self.create_grid(self.width, self.height)
        self.create_outer_wall()
        self.create_room_walls()
        self.create_room_doors()

        # Select a random initial state and goal
        self.reset_state_goal()

        # Add goal
        self.goals = list()
        self.add_goal(self.goal_pos)

        return self._get_obs()
        # return self.state, self.goal_pos

    def create_room_walls(self):
        x = self.grid.width // 2
        y = self.grid.height // 2
        self.grid.vert_wall(x, 0)
        self.grid.horz_wall(0, y)

    def create_room_doors(self):
        x_wall = self.grid.width // 2
        y_wall = self.grid.height // 2

        x_door = self.grid.width // 4
        y_door = self.grid.height // 4

        # Remove walls at door positions
        self.grid.set(x_wall, y_door, None)
        self.grid.set(x_wall, self.grid.height - 1 - y_door, None)
        self.grid.set(x_door, y_wall, None)
        self.grid.set(self.grid.width - 1 - x_door, y_wall, None)

    def _get_obs(self):
        ach_goal = self.agent_pos
        des_goal = self.goal_pos

        obs = self.grid.grid
        new_obs = []
        '''
        for object in obs:
            if type(object) == Wall:
                new_obs.append(0)
            else:
                new_obs.append(1)

        new_obs = np.array(new_obs.copy())
        new_obs = new_obs.reshape(self.width + 2, self.height + 2)
        new_obs = new_obs[1:-1, 1:-1]
        new_obs = new_obs.flatten()
        new_obs = np.append(new_obs,np.array(ach_goal))
        new_obs = np.append(new_obs, self.agent_dir)
        '''
        t = np.zeros(4)
        t[self.agent_dir] = 1
        new_obs = np.append(np.array(ach_goal), np.array(t))

        return {
            "observation": new_obs.copy(),
            "achieved_goal": np.array(ach_goal.copy()),
            "desired_goal": np.array(des_goal.copy()),
        }

    def step(self, action):
        self.step_count += 1

        reward = -1
        done = False
        info = {}
        info['is_success'] = False
        info['TimeLimit.truncated'] = False

        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == self.actions.forward:
            fwd = self.agent_pos + DIR_TO_VEC[self.agent_dir]
            if not isinstance(self.grid.get(*self.to_grid_coords(fwd)), Wall):
                self.agent_pos = fwd
        else:
            raise ValueError('Action out of bounds')

        if self.step_count >= self.max_steps:
            done = True
            info['TimeLimit.truncated'] = True
        if np.array_equal(self.agent_pos, self.goal_pos):
            done = True
            info['is_success'] = True
            reward = 0
        return self._get_obs(), reward, done, info

    def compute_reward(self, achieved_goal, goal, info):
        d = self.goal_distance(achieved_goal, goal)
        return -(d != 0).astype(np.float32)

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
