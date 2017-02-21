import numpy as np
from env.stone import Stone
import networkx as nx
import copy

class Board():

    BLACK = -1
    WHITE = 1

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

    def __init__(self, size, pieces, capstones, standing=True, flatstones=True):
        self.size = size
        self.pieces = pieces
        self.capstones = capstones
        self.height = 1
        self.standing=standing
        self.flatstones=flatstones

    def __copy__(self):
        """ get a copy of the board for hallucinating moves """
        shadow = type(self)(self.size, self.pieces, self.capstones)
        shadow.state = copy.deepcopy(self.state)
        shadow.available_pieces = copy.deepcopy(self.available_pieces)

        shadow.sides = self.sides
        shadow.standing = self.standing
        shadow.flatstones = self.flatstones
        return shadow

    def reset(self):
        self.state = np.zeros((self.size, self.size, self.height))
        self.available_pieces = {}
        self.available_pieces[Board.WHITE] = {'pieces': self.pieces, 'capstones': self.capstones}
        self.available_pieces[Board.BLACK] = {'pieces': self.pieces, 'capstones': self.capstones}

        self.sides = self.get_sides()

    def act(self, action, player):
        """ player takes an action """
        if action.get('action') == 'place':
            self.place(action, player)
        elif action.get('action') == 'move':
            self.move(action)

    def place(self, action, player):
        """ place action """
        space = action.get('to')
        piece = action.get('piece')
        top = self.get_top_index(space)
        self.state[space][top] = player * piece.value

        available = self.get_available_pieces(player)
        if piece is Stone.CAPITAL:
            available['capstones'] -= 1
        else:
            available['pieces'] -= 1

    def move(self, action):
        """ move action """
        # extract info from action
        place_from = action.get('from')
        place_to = action.get('to')
        carry = action.get('carry')
        terminal = action.get('terminal')

        values = []
        from_top = self.get_top_index(place_from)

        for idx in range(from_top - carry, from_top):
            values.append(self.state[place_from][idx])
            self.state[place_from][idx] = 0

        to_top = self.get_top_index(place_to)

        for idx, value in enumerate(values):
            # TODO FIX THIS
            # when moving, first make sure the layer exists
            # if to_top + idx == len(self.state):
            #     self.add_layer()

            self.state[place_to][to_top + idx] = value

    def get_owned_spaces(self, player, stones_types = None):
        """Return spaces owned by player
        array of tuples [(3, 2), ...]
        """
        board = self.get_top_layer()

        # determine which stones types to look for
        if stones_types is None:
            stones_types = [Stone.FLAT, Stone.STANDING, Stone.CAPITAL]
        stones_types = [x.value for x in stones_types]
        stones_types = np.array(stones_types)
        stones_types *= player

        # get matching indexes
        ix = np.in1d(board.ravel(), stones_types).reshape(board.shape)

        spaces = np.array(np.where(ix))
        return tuple((zip(*spaces)))

    def get_movement_spaces(self):
        board = self.get_top_layer()
        # todo valid movement pieces are filtered out later because of combinations,
        # todo simplify this.
        # available spaces to move
        stones = [
            Stone.EMPTY.value, 
            Stone.FLAT.value,
            -Stone.FLAT.value,
            Stone.STANDING.value,
            -Stone.STANDING.value
        ]

        ix = np.in1d(board.ravel(), stones).reshape(board.shape)
        spaces = np.array(np.where(ix))
        return tuple((zip(*spaces)))

    def can_move(self, space_from, space_to, pieces = []):

        if not space_to:
            return False

        to_piece = self.get_pieces_at_space(space_to , 1)

        # all stones can move on empty or flat stones
        if to_piece[-1] in [Stone.EMPTY, Stone.FLAT]:
            return True

        # capital stones can move on flat stones
        result = pieces[-1] == Stone.CAPITAL and to_piece == Stone.STANDING and len(pieces) == 1

        return result

    def get_pieces_at_space(self, space, num_pieces):
        idx = self.get_top_index(space)
        if idx == 0:
            return [Stone.EMPTY]

        pieces = []
        top_occupied = idx - 1
        for i in range(num_pieces):
            to_value = self.state[space][top_occupied - i]
            to_value = np.absolute(to_value)
            pieces.insert(0, Stone(to_value))

        return pieces

    def get_open_spaces(self):
        board = self.get_top_layer()
        spaces = np.array(np.where(board == 0))
        return tuple((zip(*spaces)))

    def has_open_spaces(self):
        return len(self.get_open_spaces()) > 0

    def get_top_layer(self):
        merged = []
        height = self.state.shape[-1]
        for idx in reversed(range(height)):
            layer = copy.copy(self.state[:, :, idx])
            if len(merged):
                layer[merged != 0] = merged[merged != 0]

            merged = layer

        return merged

    def get_top_index(self, space):
        for idx in range(len(self.state[space])):
            if self.state[space][idx] == 0:
                return idx

        return len(self.state)

    def is_adjacent(self, space1, space2):
        """ Returns {boolean} if two spaces are adjacent """
        # a space is adjacent if the total distance away is one
        diff = np.sum(np.absolute(np.array(space1) - np.array(space2)))
        return diff == 1

    def add_layer(self):
        self.state = np.append(
            self.state,
            np.zeros((1, self.size, self.size)),
            axis=0
        )

    def get_available_pieces(self, player):
        """ Returns {dict} available pieces for player
        {
            'pieces': 10,
            'capstones': 1
        }
        """
        return self.available_pieces.get(player)

    def get_available_piece_types(self, player):
        """ Returns all available types of pieces to place """
        num_available = self.get_available_pieces(player)
        available = []
        if num_available.get('pieces', 0):
            available.append(Stone.FLAT)
        if num_available.get('pieces', 0) and self.standing:
            available.append(Stone.STANDING)
        if num_available.get('capstones', 0):
            available.append(Stone.CAPITAL)
        return available

    def get_points(self, player, winner):

        # game tied if no winner
        if not winner:
            return 0

        pieces = self.get_available_pieces(winner)

        score = self.size ** 2
        score += pieces.get('pieces')
        score += pieces.get('capstones')

        # will make negative if player lost
        score = score * player * winner
        return score

    def get_flat_winner(self):
        """
        If no one accomplishes a road win, you can also win by controlling the most spaces with flat stones when the game ends.
        The game ends when all spaces are covered, or when one player places his last piece. 
        This is called a "flat win" or "winning the flats."
        If the flatstone score was tied it is just a tie.

        http://cheapass.com//wp-content/uploads/2016/07/Tak-Beta-Rules.pdf
        """
        white = self.get_owned_spaces(self.WHITE, stones_types = [Stone.FLAT])
        black = self.get_owned_spaces(self.BLACK, stones_types = [Stone.FLAT])

        if len(white) > len(black):
            return Board.WHITE
        elif len(white) < len(black):
            return Board.BLACK

        return 0

    def is_road_connected(self, player):
        """
        Determine if two sides of the board are connected by a road

        The object is to create a line of your pieces, called a road, connecting two opposite sides of the
        self. The road does not have to be a straight line. Each stack along the road must be topped by either
        a flatstone or a capstone in your color. Below is an example of a winning position.

        http://cheapass.com//wp-content/uploads/2016/07/Tak-Beta-Rules.pdf
        """

        # create node graph so we can check if sides are connceted
        G = nx.from_numpy_matrix(self.adjaceny_matrix(player))

        # check if North/South sides are connected
        for pos_north in self.sides.get('north'):
            for pos_south in self.sides.get('south'):
                if (nx.has_path(G, pos_north, pos_south)):
                    return True

        # check if East/West sides are connected
        for pos_west in self.sides.get('west'):
            for pos_east in self.sides.get('east'):
                if (nx.has_path(G, pos_west, pos_east)):
                    return True

        return False

    def adjaceny_matrix(self , player):
        """ adjacency matrix indicates what nodes are connected """

        adjacency = np.zeros((self.size**2, self.size**2))
        spaces = self.get_owned_spaces(player, [Stone.FLAT, Stone.CAPITAL])

        # for each space, compare against other spaces to see if they are adjacent
        for space1 in spaces:
            idx1 = self.get_node_num(space1)
            for space2 in spaces:
                if self.is_adjacent(space1, space2):
                    idx2 = self.get_node_num(space2)
                    adjacency[(idx1, idx2)] = 1

        return adjacency

    def get_sides(self):
        """ 
        spaces contained in the sides:
        north, south, east, west 
        """
        north, south, west, east = [], [], [], []
        for i in range(self.size):
            north.append(self.get_node_num((0, i)))
            south.append(self.get_node_num((self.size - 1, i)))
            west.append(self.get_node_num((i, 0)))
            east.append(self.get_node_num((i, self.size - 1)))

        return {
            'north': north, 
            'south': south, 
            'west': west,
            'east': east
        }

    def get_node_num(self, space):
        return space[0] * self.size + space[1]