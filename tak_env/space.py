
import numpy as np
import itertools as it
from tak_env.types import Stone
from tak_env import board

class ActionSpace():

    def __init__(self, **kwargs):
        self.env = kwargs.get('env')

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        valid = self.get_valid_moves(self.env.state, self.env.available_pieces, self.env.turn, self.env.continued_action)
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

    def get_valid_moves(self, state, available_pieces, turn, continued_action):
        """ Returns all current valid actions """
        if (continued_action):
            return self.get_available_next_moves(state, continued_action)
        return self.get_movements(state, turn) + self.get_placements(state, available_pieces, turn)

    def get_available_next_moves(self, state, action):
        """ 
        Returns valid continue moves
        Player must continue moving in same direction
        """
        board_size = board.get_size(state)
        next_from = action.get('to')
        next_to = self.get_next_space(board_size, action.get('from'), action.get('to'))
        next_carry_limit = action.get('carry') - 1
        return self.get_movements_from_to(state, next_from, next_to, next_carry_limit)


    def get_next_space(self, board_size, space_from, space_to):
        diff = np.array(space_to) - np.array(space_from)
        new_to = space_to + diff

        if new_to[0] not in range(board_size) or new_to[1] not in range(board_size):
            return None

        return tuple(new_to)

    def get_placements(self, state, available_pieces, turn):
        """ Returns all available piece placement actions """
        return self.get_combinations({
            'action': ['place'],
            'terminal': [True],
            'to': board.get_open_spaces(state),
            'piece': self.env.get_available_piece_types(available_pieces, turn)
        })

    def get_movements(self, state, turn):
        """ Returns all available piece movement actions """
        owned = board.get_owned_spaces(state, turn)
        available = board.get_movement_spaces(state)
        board_size = board.get_size(state)

        moves = []
        for space_owned in owned:
            pieces = board.get_pieces_at_space(state, space_owned)
            carry_limit = len(pieces)
            if carry_limit > board_size:
                carry_limit = board_size
            # find available places to move for current space
            for space_available in available:
                if board.is_adjacent(space_owned, space_available):
                    moves += self.get_movements_from_to(state, space_owned, space_available, carry_limit)

        return moves

    def get_movements_from_to(self, state, space_from, space_to, carry_limit):
        """ Returns and array of possible ways for moving from one space to another """

        combinations = self.get_combinations({
            'carry': [i for i in range(1, carry_limit + 1)],
            'terminal': [True, False],
            'from': [space_from],
            'to': [space_to],
            'action': ['move']
        })

        board_size = board.get_size(state)
        next_space = self.get_next_space(board_size, space_from, space_to)
        keep = []

        for i in combinations:

            # if terminal move and only carrying 1 piece, then the move is not valid
            if not i.get('terminal') and i.get('carry') == 1:
                continue

            pieces = board.get_pieces_at_space(state, space_from)
            pieces = pieces[:i.get('carry')]
            
            # cannot move here, not valid
            if not can_move(state, space_to, pieces):
                continue

            # if not terminal and next move is invalid then combination not valid
            can_move_next = can_move(state, next_space, pieces[-1:])
            if not i.get('terminal') and not can_move_next:
                continue

            keep.append(i)

        return keep

    def get_combinations(self, variants):
        """ Returns combinations given varients dictionary """
        import itertools as it
        varNames = sorted(variants)
        return [dict(zip(varNames, prod)) for prod in it.product(*(variants[varName] for varName in varNames))]

def can_move(state, space_to, pieces = []):

    if not space_to:
        return False

    to_piece = board.get_pieces_at_space(state, space_to)[0]
    to_stone = Stone(abs(to_piece))

    # all stones can move on empty or flat stones
    if to_stone in [Stone.EMPTY, Stone.FLAT]:
        return True

    # capital stones can move on standing stones as well
    result = pieces[0] == Stone.CAPITAL.value and to_piece == Stone.STANDING.value and len(pieces) == 1
    return result
