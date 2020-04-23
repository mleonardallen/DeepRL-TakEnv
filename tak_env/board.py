import numpy as np
from tak_env.types import Stone, Player
import copy
import itertools
from memoization import cached


"""
Board Size, along with Pieces & Capstones are configured through the environment registration

register(
    id='Tak3x3-points-v0',
    entry_point='tak_env.env:TakEnv',
    timestep_limit=200,
    kwargs={
        'board_size': 3,
        'pieces': 10,
        'capstones': 0,
        'scoring': 'points'
    }
)
"""

def get_owned_spaces(state, player, stones_types = None):
    """Return spaces owned by player
    array of tuples [(3, 2), ...]
    """

    # determine which stones types to look for
    if stones_types is None:
        stones_types = [Stone.FLAT, Stone.STANDING, Stone.CAPITAL]
    stones_types = tuple([x.value * player for x in stones_types])
    return get_spaces(state, stones_types)

def get_movement_spaces(state):
    # todo valid movement pieces are filtered out later because of combinations,
    # todo simplify this.
    # available spaces to move
    stones_types = (
        Stone.EMPTY.value, 
        Stone.FLAT.value,
        -Stone.FLAT.value,
        Stone.STANDING.value,
        -Stone.STANDING.value
    )

    return get_spaces(state, stones_types)

def get_pieces_at_space(state, space, num_pieces):

    idx = get_top_index(state, space)
    if idx == 0:
        return [Stone.EMPTY]

    pieces = []
    top_occupied = idx - 1
    for i in range(num_pieces):
        to_value = state[space][top_occupied - i]
        to_value = np.absolute(to_value)
        pieces.insert(0, Stone(to_value))

    return pieces

def get_spaces(state, stones_types):
    """
    get matching indexes
    """

    """
    [[False True False
        [True False False]
        [False False False]]
    """
    top_layer_ravel = get_top_layer(state).ravel()
    size = state.shape[0]
    ix = np.in1d(top_layer_ravel, stones_types).reshape((size, size))

    """
    Example result
    ((0,1), (1,0))
    """
    spaces = np.where(ix)
    return tuple((zip(*spaces)))

def get_open_spaces(state):
    board = get_top_layer(state)
    spaces = np.array(np.where(board == 0))
    return tuple((zip(*spaces)))

def has_open_spaces(state):
    return len(get_open_spaces(state)) > 0

def get_top_layer(state):
    merged = []
    height = state.shape[-1]
    for idx in reversed(range(height)):
        layer = copy.copy(state[:, :, idx])
        if len(merged):
            layer[merged != 0] = merged[merged != 0]

        merged = layer

    return merged

def get_top_index(state, space):
    height = state.shape[-1]
    for idx in range(height):
        if state[space][idx] == 0:
            return idx
    return len(state)

def is_adjacent(space1, space2):
    """ Returns {boolean} if two spaces are adjacent """
    diff = np.sum(np.absolute(np.array(space1) - np.array(space2)))
    return diff == 1

def add_layer(state):
    size = state.shape[0]
    # TODO do not modify in place
    return np.append(
        state,
        np.zeros((1, size, size)),
        axis=0
    )
