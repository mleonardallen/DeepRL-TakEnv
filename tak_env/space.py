
import numpy as np
import itertools as it
from tak_env.types import Stone, Direction
from tak_env import board

class ActionSpace():

    def __init__(self, **kwargs):
        self.env = kwargs.get('env')

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        state = self.env.state
        available = self.env.available_pieces
        turn = self.env.turn
        stone_types = self.env.get_available_piece_types(available, turn)

        valid = get_movements(state, turn) + \
            get_placements(state, turn, stone_types)

        return np.random.choice(valid)

    def place(self, state, action, available_pieces, player):
        """ place action """
        space = action.get('to')
        piece = action.get('piece')
        return board.put(state, space, [player * piece.value])

    def move(self, state, action):
        """ move action """
        carry = action.get('carry')
        direction = action.get('direction')
        space_from = action.get('from')
        board_size = board.get_size(state)

        # perform individual move parts
        for n in carry:
            space_to = get_next_space(board_size, space_from, direction)
            state = board.move(state, space_from, space_to, n)
            space_from = space_to

        return state

def get_placements(state, turn, stone_types):
    """ Returns all available piece placement actions """
    return get_combinations({
        'action': ['place'],
        'to': board.get_open_spaces(state),
        'piece': stone_types
    })

def get_movements(state, turn):
    """ Returns all available piece movement actions """

    carry_limit = board.get_size(state)
    spaces = board.get_owned_spaces(state, turn)

    combinations = get_combinations({
        'carry': get_carry_partitions(carry_limit),
        'from': spaces,
        'direction': [
            Direction.LEFT.value,
            Direction.RIGHT.value,
            Direction.UP.value,
            Direction.DOWN.value
        ],
        'action': ['move']
    })

    filterFn = lambda action: is_valid_move_action(state, action)
    return list(filter(filterFn, combinations))

def get_carry_partitions(carry_limit):
    carry = []
    for i in range(1, carry_limit + 1):
        carry.extend(get_partitions(i))
    # extend messes up types, so convert back to tuple
    return [x if type(x) is tuple else (x,) for x in carry]

def is_valid_move_action(state, action):
    
    space_from = action.get('from')
    carry = action.get('carry')
    direction = action.get('direction')

    # first get the pieces to be moved
    n = sum(list(carry))
    pieces = board.get_pieces_at_space(state, space_from)[:n]
    if (len(pieces) < n):
        return False

    board_size = board.get_size(state)

    # are individual move parts valid?
    for x in carry:

        space_to = get_next_space(board_size, space_from, direction)
        if space_to == None:
            return False

        if not is_valid_move_part(state, pieces[-x:], space_to):
            return False
        
        space_from = space_to
        pieces = pieces[:-x]

    return True

def is_valid_move_part(state, pieces, space_to):
    """ is the move valid accoring to the rules?
    https://cheapass.com/wp-content/uploads/2016/07/Tak-Beta-Rules.pdf
    """
    
    top_piece = board.get_pieces_at_space(state, space_to)[0]
    top_stone = Stone(abs(top_piece))

    """ Insurmountable Pieces: Neither a capstone nor a standing stone may have any piece stacked on top of it.
    These pieces can be placed and moved normally, but can’t be stacked upon.
    Therefore, it’s not legal to make a move that would place a piece atop either of these stones.
    """
    if top_stone in [Stone.EMPTY, Stone.FLAT]:
        return True

    """ Flattening Stones: The capstone can move onto any standing stone, flattening it. A standing stone can
    only be flattened by the capstone by itself, not by a taller stack with the capstone on top.
    """
    return len(pieces) == 1 \
        and abs(pieces[0]) == Stone.CAPITAL.value \
        and top_piece == Stone.STANDING.value

def get_next_space(board_size, space_from, direction):

    row, col = space_from
    if (direction == Direction.UP.value):
        row -= 1
    if (direction == Direction.DOWN.value):
        row += 1
    if (direction == Direction.LEFT.value):
        col -= 1
    if (direction == Direction.RIGHT.value):
        col += 1

    space_to = (row, col)
    if space_to[0] not in range(board_size) or space_to[1] not in range(board_size):
        return None
    return space_to

def get_combinations(variants):
    """ Returns combinations given varients dictionary """
    import itertools as it
    varNames = sorted(variants)
    return [dict(zip(varNames, prod)) for prod in it.product(*(variants[varName] for varName in varNames))]

def get_partitions(n):
    """
    partition integer
    """
    return np.unique([P for C in accel_asc(n) for P in it.permutations(C)])

def accel_asc(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]
