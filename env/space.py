from gym import Space
import numpy as np
import itertools as it
from env.stone import Stone
from env.board import Board


class ActionSpace(Space):

    def __init__(self, **kwargs):
        self.env = kwargs.get('env')

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        valid = self.get_valid_moves()
        return np.random.choice(valid)

    def get_valid_moves(self):
        return self.get_movements() + self.get_placements()

    def get_placements(self):
        return self.get_combinations({
            'action': ['place'],
            'terminal': [True],
            'to': Board.get_open_spaces(),
            'piece': self.get_available_pieces()
        })

    def get_movements(self):

        owned = Board.get_owned_spaces(self.env.turn)
        available = Board.get_movement_spaces()

        moves = []
        for space_owned in owned:

            # determine how many pieces we can carry
            carry_limit = Board.get_top_index(space_owned)
            if carry_limit > Board.size:
                carry_limit = Board.size

            for space_available in available:
                if Board.is_adjacent(space_owned, space_available):
                    moves += self.get_movements_from_to(space_owned, space_available, carry_limit)

        return moves

    def get_movements_from_to(self, space_from, space_to, carry_limit):
        combinations = self.get_combinations({
            'carry': [i for i in range(1, carry_limit + 1)],
            'terminal': [True, False],
            'from': [space_from],
            'to': [space_to],
            'action': ['move']
        })

        # all moves that carry one piece are terminal
        for i in combinations:
            if not i.get('terminal') and i.get('carry') == 1:
                combinations.remove(i)

        return combinations

    def get_available_pieces(self):
        num_available = self.env.get_available()
        available = []
        if num_available.get('pieces', 0):
            available.append(Stone.FLAT)
            available.append(Stone.STANDING)
        if num_available.get('captones', 0):
            available.append(Stone.CAPITAL)
        return available

    def get_combinations(self, variants):
        import itertools as it
        varNames = sorted(variants)
        return [dict(zip(varNames, prod)) for prod in it.product(*(variants[varName] for varName in varNames))]


    def __repr__(self):
        return "TakActionSpace" + str(self.num_discrete_space)