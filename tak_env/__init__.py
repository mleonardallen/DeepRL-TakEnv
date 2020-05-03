from gym.envs.registration import register

# SCORING BY POINTS

register(
    id='Tak3x3-points-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 3,
        'pieces': 10,
        'capstones': 0,
        'scoring': 'points'
    }
)

register(
    id='Tak4x4-points-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 4,
        'pieces': 15,
        'capstones': 0,
        'scoring': 'points'
    }
)

register(
    id='Tak5x5-points-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 5,
        'pieces': 21,
        'capstones': 1,
        'scoring': 'points'
    }
)

register(
    id='Tak6x6-points-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 6,
        'pieces': 30,
        'capstones': 1,
        'scoring': 'points'
    }
)

register(
    id='Tak7x7-points-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 7,
        'pieces': 40,
        'capstones': 1,
        'scoring': 'points'
    }
)

register(
    id='Tak7x7-2cap-points-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 7,
        'pieces': 40,
        'capstones': 2,
        'scoring': 'points'
    }
)

register(
    id='Tak8x8-points-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 8,
        'pieces': 50,
        'capstones': 2,
        'scoring': 'points'
    }
)

# SCORING BY WINS

register(
    id='Tak3x3-wins-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 3,
        'pieces': 10,
        'capstones': 0,
        'scoring': 'wins'
    }
)

register(
    id='Tak4x4-wins-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 4,
        'pieces': 15,
        'capstones': 0,
        'scoring': 'wins'
    }
)

register(
    id='Tak5x5-wins-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 5,
        'pieces': 21,
        'capstones': 1,
        'scoring': 'wins'
    }
)

register(
    id='Tak6x6-wins-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 6,
        'pieces': 30,
        'capstones': 1,
        'scoring': 'wins'
    }
)

register(
    id='Tak7x7-wins-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 7,
        'pieces': 40,
        'capstones': 1,
        'scoring': 'wins'
    }
)

register(
    id='Tak7x7-2cap-wins-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 7,
        'pieces': 40,
        'capstones': 2,
        'scoring': 'wins'
    }
)

register(
    id='Tak8x8-wins-v0',
    entry_point='tak_env.env:TakEnv',
    kwargs={
        'board_size': 8,
        'pieces': 50,
        'capstones': 2,
        'scoring': 'wins'
    }
)