
import numpy as np
import itertools as it
from tak_env.types import Stone
from tak_env import board

class ActionSpace():

    def __init__(self, **kwargs):
        self.env = kwargs.get('env')

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        state = self.env.state
        available = self.env.available_pieces
        turn = self.env.turn
        continued_action = self.env.continued_action
        stone_types = self.env.get_available_piece_types(available, turn)

        valid = get_valid_moves(state, stone_types, turn, continued_action)
        return np.random.choice(valid)

    def place(self, state, action, available_pieces, player):
        """ place action """
        space = action.get('to')
        piece = action.get('piece')
        return board.put(state, space, [player * piece.value])

    def move(self, state, action):
        """ move action """
        place_from = action.get('from')
        place_to = action.get('to')
        carry = action.get('carry')

        pieces = board.get_pieces_at_space(state, place_from)[:carry]
        state = board.remove(state, place_from, carry)
        state = board.put(state, place_to, pieces)
        return state

def get_valid_moves(state, stone_types, turn, continued_action):
    """ Returns all current valid actions """
    if (continued_action):
        return get_available_next_moves(state, continued_action)
    return get_movements(state, turn) + get_placements(state, stone_types, turn)

def get_available_next_moves(state, action):
    """ 
    Returns valid continue moves
    Player must continue moving in same direction
    """
    board_size = board.get_size(state)
    next_from = action.get('to')
    next_to = get_next_space(board_size, action.get('from'), action.get('to'))
    next_carry_limit = action.get('carry') - 1
    return get_movements_from_to(state, next_from, next_to, next_carry_limit)

def get_placements(state, stone_types, turn):
    """ Returns all available piece placement actions """
    return get_combinations({
        'action': ['place'],
        'terminal': [True],
        'to': board.get_open_spaces(state),
        'piece': stone_types
    })

def get_movements(state, turn):
    """ Returns all available piece movement actions """
    owned = board.get_owned_spaces(state, turn)
    available = board.get_movement_spaces(state)

    moves = []
    board_size = board.get_size(state)
    
    for space_owned in owned:
        pieces = board.get_pieces_at_space(state, space_owned)
        carry_limit = len(pieces)
        if carry_limit > board_size:
            carry_limit = board_size
        # find available places to move for current space
        for space_available in available:
            if board.is_adjacent(space_owned, space_available):
                moves += get_movements_from_to(state, space_owned, space_available, carry_limit)

    return moves

def get_movements_from_to(state, space_from, space_to, carry_limit):
    """ Returns and array of possible ways for moving from one space to another """
    combinations = get_combinations({
        'carry': [i for i in range(1, carry_limit + 1)],
        'terminal': [True, False],
        'from': [space_from],
        'to': [space_to],
        'action': ['move']
    })
    filterFn = lambda action: is_valid_move_action(state, space_from, space_to, action)
    return list(filter(filterFn, combinations))

def get_combinations(variants):
    """ Returns combinations given varients dictionary """
    import itertools as it
    varNames = sorted(variants)
    return [dict(zip(varNames, prod)) for prod in it.product(*(variants[varName] for varName in varNames))]


def is_valid_move_action(state, space_from, space_to, action):
    # if terminal move and only carrying 1 piece, then the move is not valid
    # artifact from generating all possible combinations
    if not action.get('terminal') and action.get('carry') == 1:
        return False

    pieces = board.get_pieces_at_space(state, space_from)
    
    # cannot move here, not valid
    if not can_move(state, space_to, pieces[:action.get('carry')]):
        return False

    # if not terminal AND next move is invalid then combination not valid
    board_size = board.get_size(state)
    next_space = get_next_space(board_size, space_from, space_to)

    # if the top piece can move, 
    # then there is at least one more move
    # top piece can be capital stone
    can_move_next = can_move(state, next_space, pieces[:1])
    if not action.get('terminal') and not can_move_next:
        return False

    return True

def can_move(state, space_to, pieces):

    # no space to,
    # TODO is it possible??
    if not space_to:
        return False

    # no pieces to move
    if Stone(abs(pieces[0])) == Stone.EMPTY:
        return False

    to_piece = board.get_pieces_at_space(state, space_to)[0]
    to_stone = Stone(abs(to_piece))

    # all stones can move on empty or flat stones
    if to_stone in [Stone.EMPTY, Stone.FLAT]:
        return True

    # capital stones can move on standing stones as well
    result = pieces[0] == Stone.CAPITAL.value \
        and to_piece == Stone.STANDING.value \
        and len(pieces) == 1
    return result

def get_next_space(board_size, space_from, space_to):
    diff = np.array(space_to) - np.array(space_from)
    new_to = space_to + diff

    if new_to[0] not in range(board_size) or new_to[1] not in range(board_size):
        return None

    return tuple(new_to)
