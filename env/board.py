import numpy as np
from env.stone import Stone
import networkx as nx
import copy

class Board():

    BLACK = -1
    WHITE = 1
    state = {}
    available_pieces = {}

    """
    Board Size, along with Pieces & Capstones are configured through the environment registration

    register(
        id='Tak3x3-points-v0',
        entry_point='env.tak:TakEnv',
        timestep_limit=200,
        kwargs={
            'board_size': 3,
            'pieces': 10,
            'capstones': 0,
            'scoring': 'points'
        }
    )
    """

    size = None
    capstones = None
    pieces = None

    @staticmethod
    def reset():
        Board.state = np.zeros((1, Board.size, Board.size))
        Board.available_pieces[Board.WHITE] = {'pieces': Board.pieces, 'capstones': Board.capstones}
        Board.available_pieces[Board.BLACK] = {'pieces': Board.pieces, 'capstones': Board.capstones}
        Board.sides = Board.get_sides()

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
    def is_road_connected(player):
        """
        Determine if two sides of the board are connected by a road

        The object is to create a line of your pieces, called a road, connecting two opposite sides of the
        board. The road does not have to be a straight line. Each stack along the road must be topped by either
        a flatstone or a capstone in your color. Below is an example of a winning position.

        http://cheapass.com//wp-content/uploads/2016/07/Tak-Beta-Rules.pdf
        """

        # create node graph so we can check if sides are connceted
        G = nx.from_numpy_matrix(Board.adjaceny_matrix(player))

        # check if North/South sides are connected
        for pos_north in Board.sides.get('north'):
            for pos_south in Board.sides.get('south'):
                if (nx.has_path(G, pos_north, pos_south)):
                    return True

        # check if East/West sides are connected
        for pos_west in Board.sides.get('west'):
            for pos_east in Board.sides.get('east'):
                if (nx.has_path(G, pos_west, pos_east)):
                    return True

        return False

    @staticmethod
    def adjaceny_matrix(player):
        """ adjacency matrix indicates what nodes are connected """

        adjacency = np.zeros((Board.size**2, Board.size**2))
        spaces = Board.get_owned_spaces(player, stones_types='win')

        # for each space, compare against other spaces to see if they are adjacent
        for space1 in spaces:
            idx1 = Board.get_node_num(space1)
            for space2 in spaces:
                if Board.is_adjacent(space1, space2):
                    idx2 = Board.get_node_num(space2)
                    adjacency[(idx1, idx2)] = 1

        return adjacency

    @staticmethod
    def get_sides():
        """ 
        spaces contained in the sides:
        north, south, east, west 
        """
        north, south, west, east = [], [], [], []
        for i in range(Board.size):
            north.append(Board.get_node_num((0, i)))
            south.append(Board.get_node_num((Board.size - 1, i)))
            west.append(Board.get_node_num((i, 0)))
            east.append(Board.get_node_num((i, Board.size - 1)))

        return {
            'north': north, 
            'south': south, 
            'west': west,
            'east': east
        }

    @staticmethod
    def get_node_num(space):
        return space[0] * Board.size + space[1]