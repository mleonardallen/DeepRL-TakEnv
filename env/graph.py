"""
Tak - A Beautiful Game
Determine if two sides of the board are connected by a road
"""

import networkx as nx
import numpy as np

class TakGraph():

    def __init__(self, board_size):

        self.board_size = board_size
        self.sides = self.get_sides()

    def is_connected(self, board, player):

        # Create graph so we can check if sides are conncetedP
        G = nx.from_numpy_matrix(self.adjaceny_matrix(board, player))

        # Check if Top/Bottom sides are connected
        for pos_top in self.sides.get('top'):
            for pos_bottom in self.sides.get('bottom'):
                if (nx.has_path(G, pos_top, pos_bottom)):
                    return True

        # Check if Left/Right sides are connected
        for pos_left in self.sides.get('left'):
            for pos_right in self.sides.get('right'):
                if (nx.has_path(G, pos_left, pos_right)):
                    return True

        return False

    def adjaceny_matrix(self, board, player):

        adjacency = np.zeros((self.board_size**2, self.board_size**2))

        # spaces owned by player
        spaces = np.array(np.where(board == player))
        spaces = tuple((zip(*spaces)))

        # for each space, compare against other spaces to see if they are adjacent
        for space1 in spaces:
            idx1 = self.get_node_num(space1)
            for space2 in spaces:
                # a space is adjacent if the total distance away is one
                diff = np.sum(np.absolute(np.array(space1) - np.array(space2)))
                if diff == 1: #adjacent
                    idx2 = self.get_node_num(space2)
                    adjacency[(idx1, idx2)] = 1

        return adjacency

    def get_sides(self):
        top, bottom, left, right = [], [], [], []
        for i in range(self.board_size):
            top.append(self.get_node_num((0, i)))
            bottom.append(self.get_node_num((self.board_size - 1, i)))
            left.append(self.get_node_num((i, 0)))
            right.append(self.get_node_num((i, self.board_size - 1)))

        return {
            'top': top, 
            'bottom': bottom, 
            'left': left,
            'right': right
        }

    def get_node_num(self, space):
        return space[0] * self.board_size + space[1]