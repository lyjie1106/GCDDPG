from gym.envs.registration import register

register(
    id='ModifiedFourRoomEnv-v0',
    entry_point='baseline.env.ModifiedFourRoomEnv:ModifiedFourRoomEnv'
)
register(
    id='ModifiedEmptyRoomEnv-v0',
    entry_point='baseline.env.ModifiedEmptyRoomEnv:ModifiedEmptyRoomEnv'
)
