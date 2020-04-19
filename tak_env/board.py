import numpy as np
from tak_env.types import Stone, Player
import networkx as nx
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

def act(state, action, available_pieces, player):
    """ player takes an action """
    if action.get('action') == 'place':
        place(state, action, available_pieces, player)
    elif action.get('action') == 'move':
        move(state, action)

def place(state, action, available_pieces, player):
    """ place action """
    space = action.get('to')
    piece = action.get('piece')
    top = get_top_index(state, space)
    # TODO do not modify in place
    state[space][top] = player * piece.value
    # TODO do not modify in place
    available_pieces = available_pieces.get(player)
    if piece is Stone.CAPITAL:
        available_pieces['capstones'] -= 1
    else:
        available_pieces['pieces'] -= 1

def move(state, action):
    """ move action """
    # extract info from action
    place_from = action.get('from')
    place_to = action.get('to')
    carry = action.get('carry')

    values = []
    from_top = get_top_index(state, place_from)

    # TODO do not modify in place
    for idx in range(from_top - carry, from_top):
        values.append(state[place_from][idx])
        state[place_from][idx] = 0

    to_top = get_top_index(state, place_to)

    for idx, value in enumerate(values):
        # TODO FIX THIS
        # when moving, first make sure the layer exists
        # if to_top + idx == len(state):
        #     add_layer(state)

        state[place_to][to_top + idx] = value

def get_owned_spaces(state, player, stones_types = None):
    """Return spaces owned by player
    array of tuples [(3, 2), ...]
    """

    # determine which stones types to look for
    if stones_types is None:
        stones_types = [Stone.FLAT, Stone.STANDING, Stone.CAPITAL]
    stones_types = [x.value for x in stones_types]
    stones_types = np.array(stones_types)
    stones_types *= player

    return get_spaces(state, stones_types)

def get_movement_spaces(state):
    # todo valid movement pieces are filtered out later because of combinations,
    # todo simplify this.
    # available spaces to move
    stones_types = [
        Stone.EMPTY.value, 
        Stone.FLAT.value,
        -Stone.FLAT.value,
        Stone.STANDING.value,
        -Stone.STANDING.value
    ]

    return get_spaces(state, stones_types)

def can_move(state, space_from, space_to, pieces = []):

    if not space_to:
        return False

    to_piece = get_pieces_at_space(state, space_to , 1)

    # all stones can move on empty or flat stones
    if to_piece[-1] in [Stone.EMPTY, Stone.FLAT]:
        return True

    # capital stones can move on flat stones
    result = pieces[-1] == Stone.CAPITAL and to_piece == Stone.STANDING and len(pieces) == 1

    return result

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

def get_available_piece_types(available_pieces, player):
    """ Returns all available types of pieces to place """
    num_available = available_pieces.get(player)
    available = []
    if num_available.get('pieces', 0):
        available.append(Stone.FLAT)
    if num_available.get('pieces', 0):
        available.append(Stone.STANDING)
    if num_available.get('capstones', 0):
        available.append(Stone.CAPITAL)
    return available

def get_points(size, available_pieces, player, winner):

    # game tied if no winner
    if not winner:
        return 0

    pieces = available_pieces.get(winner)
    score = size ** 2
    score += pieces.get('pieces')
    score += pieces.get('capstones')

    # will make negative if player lost
    score = score * player * winner
    return score

def get_flat_winner(state):
    """
    If no one accomplishes a road win, you can also win by controlling the most spaces with flat stones when the game ends.
    The game ends when all spaces are covered, or when one player places his last piece. 
    This is called a "flat win" or "winning the flats."
    If the flatstone score was tied it is just a tie.

    http://cheapass.com//wp-content/uploads/2016/07/Tak-Beta-Rules.pdf
    """
    white = get_owned_spaces(state, Player.WHITE.value, stones_types = [Stone.FLAT])
    black = get_owned_spaces(state, Player.BLACK.value, stones_types = [Stone.FLAT])

    if len(white) > len(black):
        return Player.WHITE.value
    elif len(white) < len(black):
        return Player.BLACK.value

    return 0

def is_road_connected(state, player):
    """
    Determine if two sides of the board are connected by a road

    The object is to create a line of your pieces, called a road, connecting two opposite sides.
    The road does not have to be a straight line. 
    Each stack along the road must be topped by either a flatstone or a capstone in your color.

    http://cheapass.com//wp-content/uploads/2016/07/Tak-Beta-Rules.pdf
    """

    # create node graph so we can check if sides are connceted
    G = nx.from_numpy_matrix(get_adjacency_matrix(state, player))
    sides = get_sides(state)
    # check if North/South sides are connected
    for pos_north in sides.get('north'):
        for pos_south in sides.get('south'):
            if (nx.has_path(G, pos_north, pos_south)):
                return True

    # check if East/West sides are connected
    for pos_west in sides.get('west'):
        for pos_east in sides.get('east'):
            if (nx.has_path(G, pos_west, pos_east)):
                return True

    return False

def get_adjacency_matrix(state, player):
    """ adjacency matrix indicates what nodes are connected """
    size = state.shape[0]
    adjacency = np.zeros((size**2, size**2))
    spaces = get_owned_spaces(state, player, [Stone.FLAT, Stone.CAPITAL])

    # for each space, compare against other spaces to see if they are adjacent
    for space1 in spaces:
        idx1 = get_node_num(size, space1)
        for space2 in spaces:
            if is_adjacent(space1, space2):
                idx2 = get_node_num(size, space2)
                adjacency[(idx1, idx2)] = 1

    return adjacency

def get_sides(state):
    """ 
    spaces contained in the sides:
    north, south, east, west 
    """
    north, south, west, east = [], [], [], []
    size = state.shape[0]
    for i in range(size):
        north.append(get_node_num(size, (0, i)))
        south.append(get_node_num(size, (size - 1, i)))
        west.append(get_node_num(size, (i, 0)))
        east.append(get_node_num(size, (i, size - 1)))

    return {
        'north': north, 
        'south': south, 
        'west': west,
        'east': east
    }

def get_node_num(size, space):
    return space[0] * size + space[1]
