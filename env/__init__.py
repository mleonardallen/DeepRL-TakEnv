from gym.envs.registration import register

register(
    id='Tak3x3-v0',
    entry_point='env.tak:TakEnv',
    timestep_limit=200,
    reward_threshold=25.0,
    kwargs={
        'board_size': 3
    }
)

register(
    id='Tak4x4-v0',
    entry_point='env.tak:TakEnv',
    timestep_limit=200,
    reward_threshold=25.0,
    kwargs={
        'board_size': 4
    }
)

register(
    id='Tak5x5-v0',
    entry_point='env.tak:TakEnv',
    timestep_limit=200,
    reward_threshold=25.0,
    kwargs={
        'board_size': 5
    }
)

register(
    id='Tak6x6-v0',
    entry_point='env.tak:TakEnv',
    timestep_limit=200,
    reward_threshold=25.0,
    kwargs={
        'board_size': 6
    }
)

register(
    id='Tak7x7-v0',
    entry_point='env.tak:TakEnv',
    timestep_limit=200,
    reward_threshold=25.0,
    kwargs={
        'board_size': 7
    }
)

register(
    id='Tak8x8-v0',
    entry_point='env.tak:TakEnv',
    timestep_limit=200,
    reward_threshold=25.0,
    kwargs={
        'board_size': 8
    }
)