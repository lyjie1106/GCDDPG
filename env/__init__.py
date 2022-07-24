from gym.envs.registration import register

register(
    id='ModifiedFourRoomEnv-v0',
    entry_point='env.ModifiedFourRoomEnv:ModifiedFourRoomEnv'
)