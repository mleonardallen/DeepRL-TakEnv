import numpy as np
from env.stone import Stone
import copy

class Board():

    BLACK = -1
    WHITE = 1

    CONFIG = {
        3: {'pieces': 10, 'capstones': 0},
        4: {'pieces': 15, 'capstones': 0},
        5: {'pieces': 21, 'capstones': 1},
        6: {'pieces': 30, 'capstones': 1},
        7: {'pieces': 40, 'capstones': 2},
        8: {'pieces': 50, 'capstones': 2},
    }

    state = {}
    available_pieces = {}
    size = None

    @staticmethod
    def reset():
        Board.state = np.zeros((1, Board.size, Board.size))
        Board.available_pieces[Board.WHITE] = copy.copy(Board.CONFIG.get(Board.size))
        Board.available_pieces[Board.BLACK] = copy.copy(Board.CONFIG.get(Board.size))

    @staticmethod
    def place(action, player):
        space = action.get('to')
        piece = action.get('piece')
        top = Board.get_top_index(space)
        Board.state[top][space] = player * piece.value

        available = Board.get_available_pieces(player)
        if piece is Stone.CAPITAL:
            available['capstones'] -= 1
        else:
            available['pieces'] -= 1

    @staticmethod
    def move(action):
        # extract info from action
        place_from = action.get('from')
        place_to = action.get('to')
        carry = action.get('carry')
        terminal = action.get('terminal')

        values = []
        from_top = Board.get_top_index(place_from)

        for idx in range(from_top - carry, from_top):
            values.append(Board.state[idx][place_from])
            Board.state[idx][place_from] = 0

        to_top = Board.get_top_index(place_to)

        for idx, value in enumerate(values):
            # when moving, first make sure the layer exists
            if to_top + idx == len(Board.state):
                Board.add_layer()

            Board.state[to_top + idx][place_to] = value

    @staticmethod
    def get_owned_spaces(player, stones_types='all'):
        board = Board.get_top_layer()

        # determine which stones to look for
        stones = [Stone.FLAT.value, Stone.STANDING.value, Stone.CAPITAL.value]
        if stones_types == 'win':
            stones = [Stone.FLAT.value, Stone.CAPITAL.value]
        if stones_types == 'flat':
            stones = [Stone.FLAT.value]

        stones = np.array(stones)
        stones *= player

        # get matching indexes
        ix = np.in1d(board.ravel(), stones).reshape(board.shape)

        spaces = np.array(np.where(ix))
        return tuple((zip(*spaces)))

    @staticmethod
    def get_movement_spaces(captital=False):
        board = Board.get_top_layer()
        # available spaces to move
        stones = [0, Stone.FLAT.value, -Stone.FLAT.value]
        # capital stone can move on top of standing stones
        if captital:
            stones = [
                0, 
                Stone.FLAT.value,
                -Stone.FLAT.value,
                Stone.STANDING.value,
                -Stone.STANDING.value
            ]

        ix = np.in1d(board.ravel(), stones).reshape(board.shape)
        spaces = np.array(np.where(ix))
        return tuple((zip(*spaces)))

    @staticmethod
    def get_open_spaces():
        board = Board.get_top_layer()
        spaces = np.array(np.where(board == 0))
        return tuple((zip(*spaces)))

    @staticmethod
    def has_open_spaces():
        return len(Board.get_open_spaces()) > 0

    @staticmethod
    def get_top_layer():
        merged = []
        for idx in reversed(range(len(Board.state))):
            layer = copy.copy(Board.state[idx])
            if len(merged):
                layer[merged != 0] = merged[merged != 0]

            merged = layer
        return merged

    @staticmethod
    def get_top_index(space):
        for idx in range(len(Board.state)):
            if Board.state[idx][space] == 0:
                return idx

        return len(Board.state)

    @staticmethod
    def is_adjacent(space1, space2):
        """ Determine if two spaces are adjacent """
        # a space is adjacent if the total distance away is one
        diff = np.sum(np.absolute(np.array(space1) - np.array(space2)))
        return diff == 1

    @staticmethod
    def add_layer():
        Board.state = np.append(
            Board.state,
            np.zeros((1, Board.size, Board.size)),
            axis=0
        )

    @staticmethod
    def get_available_pieces(player):
        # keep track of available pieces
        return Board.available_pieces.get(player)

    @staticmethod
    def get_available_piece_types(player):
        """ Returns all available types of pieces to place """
        num_available = Board.get_available_pieces(player)
        available = []
        if num_available.get('pieces', 0):
            available.append(Stone.FLAT)
            available.append(Stone.STANDING)
        if num_available.get('captones', 0):
            available.append(Stone.CAPITAL)
        return available

    @staticmethod
    def get_flat_winner():
        """
        If no one accomplishes a road win, you can also win by controlling the most spaces with flat stones when the game ends.
        The game ends when all spaces are covered, or when one player places his last piece. 
        This is called a "flat win" or "winning the flats."
        If the flatstone score was tied it is just a tie.

        http://cheapass.com//wp-content/uploads/2016/07/Tak-Beta-Rules.pdf
        """
        white = Board.get_owned_spaces(Board.WHITE, stones_types='flat')
        black = Board.get_owned_spaces(Board.BLACK, stones_types='flat')

        if len(white) > len(black):
            return Board.WHITE
        elif len(white) < len(black):
            return Board.BLACK

        return 0

    @staticmethod
    def get_points(player, winner):

        if not winner:
            return 0

        pieces = Board.get_available_pieces(winner)

        score = Board.size ** 2
        score += pieces.get('pieces')
        score += pieces.get('capstones')

        # will make negative if player lost
        score = score * player * winner
        return score
