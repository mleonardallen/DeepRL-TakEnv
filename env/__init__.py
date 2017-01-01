from gym.envs.registration import register

register(
    id='Tak-v0',
    entry_point='env.tak:TakEnv',
    timestep_limit=200,
    reward_threshold=25.0
)
