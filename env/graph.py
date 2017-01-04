import networkx as nx
import numpy as np
from env.board import Board
from env.stone import Stone

class ConnectionGraph():
    """
    Determine if two sides of the board are connected by a road
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # @todo listen for event of board reset
        self.sides = self.get_sides()

    def is_connected(self, player):
        """
        Is connected if...
        node on north side connected to south side
        node on east side connected to west side
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

    def adjaceny_matrix(self, player):
        """ adjacency matrix indicates what nodes are connected """

        adjacency = np.zeros((Board.size**2, Board.size**2))
        spaces = Board.get_owned_spaces(player, only_towards_win=True)

        # for each space, compare against other spaces to see if they are adjacent
        for space1 in spaces:
            idx1 = self.get_node_num(space1)
            for space2 in spaces:
                if Board.is_adjacent(space1, space2):
                    idx2 = self.get_node_num(space2)
                    adjacency[(idx1, idx2)] = 1

        return adjacency

    def get_sides(self):
        """ 
        spaces contained in the sides:
        north, south, east, west 
        """

        north, south, west, east = [], [], [], []
        for i in range(Board.size):
            north.append(self.get_node_num((0, i)))
            south.append(self.get_node_num((Board.size - 1, i)))
            west.append(self.get_node_num((i, 0)))
            east.append(self.get_node_num((i, Board.size - 1)))

        return {
            'north': north, 
            'south': south, 
            'west': west,
            'east': east
        }

    def get_node_num(self, space):
        return space[0] * Board.size + space[1]