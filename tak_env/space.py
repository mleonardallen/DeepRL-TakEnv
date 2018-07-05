
import numpy as np
import itertools as it
from tak_env.stone import Stone

class ActionSpace():

    def __init__(self, **kwargs):
        self.env = kwargs.get('env')

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        valid = self.get_valid_moves()
        return np.random.choice(valid)

    def get_valid_moves(self):
        """ Returns all current valid actions """
        if (self.env.continued_action):
            return self.get_available_next_moves(self.env.continued_action)

        return self.get_movements() + self.get_placements()

    def get_available_next_moves(self, action):
        """ 
        Returns valid continue moves
        Player must continue moving in same direction
        """
        action = self.env.continued_action
        next_from = action.get('to')
        next_to = self.get_next_space(action.get('from'), action.get('to'))
        next_carry_limit = action.get('carry') - 1
        return self.get_movements_from_to(next_from, next_to, next_carry_limit)


    def get_next_space(self, space_from, space_to):
        diff = np.array(space_to) - np.array(space_from)
        new_to = space_to + diff

        if new_to[0] not in range(self.env.board.size) or new_to[1] not in range(self.env.board.size):
            return None

        return tuple(new_to)

    def get_placements(self):
        """ Returns all available piece placement actions """
        return self.get_combinations({
            'action': ['place'],
            'terminal': [True],
            'to': self.env.board.get_open_spaces(),
            'piece': self.env.board.get_available_piece_types(self.env.turn)
        })

    def get_movements(self):
        """ Returns all available piece movement actions """
        owned = self.env.board.get_owned_spaces(self.env.turn)
        available = self.env.board.get_movement_spaces()

        moves = []
        for space_owned in owned:
            # determine how many pieces we can carry
            carry_limit = self.env.board.get_top_index(space_owned)
            if carry_limit > self.env.board.size:
                carry_limit = self.env.board.size
            # find available places to move for current space
            for space_available in available:
                if self.env.board.is_adjacent(space_owned, space_available):
                    moves += self.get_movements_from_to(space_owned, space_available, carry_limit)

        return moves

    def get_movements_from_to(self, space_from, space_to, carry_limit):
        """ Returns and array of possible ways for moving from one space to another """

        combinations = self.get_combinations({
            'carry': [i for i in range(1, carry_limit + 1)],
            'terminal': [True, False],
            'from': [space_from],
            'to': [space_to],
            'action': ['move']
        })

        next_space = self.get_next_space(space_from, space_to)
        keep = []

        for i in combinations:

            # if terminal move and only carrying 1 piece, then the move is not valid
            if not i.get('terminal') and i.get('carry') == 1:
                continue

            pieces = self.env.board.get_pieces_at_space(space_from, i.get('carry'))
            can_move = self.env.board.can_move(space_from, space_to, pieces)

            # cannot move here, not valid
            if not can_move:
                continue

            # if not terminal and next move is invalid then combination not valid
            can_move_next = self.env.board.can_move(space_to, next_space, pieces[-1:])
            if not i.get('terminal') and not can_move_next:
                continue

            keep.append(i)

        return keep

    def get_combinations(self, variants):
        """ Returns combinations given varients dictionary """
        import itertools as it
        varNames = sorted(variants)
        return [dict(zip(varNames, prod)) for prod in it.product(*(variants[varName] for varName in varNames))]


    def __repr__(self):
        return "TakActionSpace" + str(self.num_discrete_space)