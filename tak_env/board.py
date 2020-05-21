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
    stones_types = [x.value * player for x in stones_types]
    return get_matching_spaces(state, stones_types)

def get_pieces_at_space(state, space):
    top_idx = get_top_index(state, space)
    column = state[:,space[0],space[1]]
    if get_height(state) == top_idx + 1:
        return column[-1:]

    top_occupied = top_idx + 1
    return column[top_occupied:]

def get_matching_spaces(state, stones_types):
    """
    get matching indexes
    """

    """
    [[False True False
        [True False False]
        [False False False]]
    """
    top_layer_ravel = get_top_layer(state).ravel()
    size = get_size(state)
    ix = np.in1d(top_layer_ravel, stones_types).reshape((size, size))

    """
    Example result
    [(0,1), (1,0)]
    """
    spaces = np.where(ix)
    return list(zip(*spaces))

def get_open_spaces(state):
    board = get_top_layer(state)
    spaces = np.array(np.where(board == 0))
    return list(zip(*spaces))

def has_open_spaces(state):
    return len(get_open_spaces(state)) > 0

def get_top_layer(state):
    merged = []
    height = get_height(state)
    for idx in range(height):
        layer = copy.copy(state[idx, :, :])
        if len(merged):
            layer[merged != 0] = merged[merged != 0]

        merged = layer
    return merged

def get_height(state):
    return state.shape[0]

def get_size(state):
    return state.shape[-1]

def get_top_index(state, space):
    """ get top available index """
    height = get_height(state)
    for idx in reversed(range(height)):
        if state[idx][space] == 0:
            return idx
    return -1

def is_adjacent(space1, space2):
    """ Returns {boolean} if two spaces are adjacent """
    diff = np.sum(np.absolute(np.array(space1) - np.array(space2)))
    return diff == 1

# manipulation of board

def move(state, space_from, space_to, n):
    state = np.copy(state)
    pieces = get_pieces_at_space(state, space_from)[:n]
    state = remove(state, space_from, n)
    state = put(state, space_to, pieces)
    return state

def remove(state, space, n):
    state = np.copy(state)
    idx = get_top_index(state, space)
    state[:,space[0],space[1]][idx+1:idx+1+n] = 0
    return state

def put(state, space, pieces):
    state = np.copy(state)
    to_top = get_top_index(state, space)
    
    # check if standing stone is flattened
    is_capstone = len(pieces) == 1 and abs(pieces[0]) == Stone.CAPITAL.value
    top_piece = get_pieces_at_space(state, space)[0]
    top_is_standing = Stone(abs(top_piece)).value == Stone.STANDING.value
    should_flatten = is_capstone and top_is_standing
    if should_flatten:
        # -2/2 = -1, 2/2 = 1
        state[to_top+1][space] /= 2

    for idx, value in enumerate(reversed(pieces)):
        # when moving, first make sure the layer exists
        place_at = to_top - idx
        if place_at < 0:
            state = add_layer(state)
            place_at = 0
        state[place_at][space] = value
    return state

def add_layer(state, n=1):
    size = get_size(state)
    state = np.copy(state)
    return np.insert(
        state,
        0,
        np.zeros((n, size, size)),
        axis=0
    )
