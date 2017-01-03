import numpy as np
from env.stone import Stone
import copy

class Board():

    MAX_HEIGHT = 1

    state = {}
    size = 4

    @staticmethod
    def reset():
        Board.state = np.zeros((Board.MAX_HEIGHT, Board.size, Board.size))

    @staticmethod
    def place(action, player):
        space = action.get('to')
        piece = action.get('piece')
        top = Board.get_top_index(space)
        Board.state[top][space] = player * piece.value

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
    def get_owned_spaces(player, only_towards_win=False):
        board = Board.get_top_layer()

        # determine which stones to look for
        stones = [Stone.FLAT.value, Stone.STANDING.value, Stone.CAPITAL.value]
        if only_towards_win:
            stones = [Stone.FLAT.value, Stone.CAPITAL.value]
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