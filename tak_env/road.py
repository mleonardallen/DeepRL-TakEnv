import tak_env.board as Board
import numpy as np
import networkx as nx

def is_connected(size, spaces):
    """
    Determine if two sides of the board are connected by a road

    The object is to create a line of your pieces, called a road, connecting two opposite sides.
    The road does not have to be a straight line. 
    Each stack along the road must be topped by either a flatstone or a capstone in your color.

    http://cheapass.com//wp-content/uploads/2016/07/Tak-Beta-Rules.pdf
    """

    # create node graph so we can check if sides are connceted
    matrix = get_adjacency_matrix(size, spaces)
    sides = get_sides(size)

    G = nx.from_numpy_matrix(matrix)
    # check if North/South sides are connected
    for pos_north in sides.get('north'):
        for pos_south in sides.get('south'):
            if (nx.has_path(G, pos_north, pos_south)):
                return True

    # check if East/West sides are connected
    for pos_west in sides.get('west'):
        for pos_east in sides.get('east'):
            if (nx.has_path(G, pos_west, pos_east)):
                return True

    return False

def get_adjacency_matrix(size, spaces):
    """ adjacency matrix indicates what nodes are connected """
    adjacency = np.zeros((size**2, size**2))
    
    # for each space, compare against other spaces to see if they are adjacent
    for space1 in spaces:
        idx1 = get_node_num(size, space1)
        for space2 in spaces:
            if Board.is_adjacent(space1, space2):
                idx2 = get_node_num(size, space2)
                adjacency[(idx1, idx2)] = 1

    return adjacency

def get_sides(size):
    """ 
    spaces contained in the sides:
    north, south, east, west 
    """
    north, south, west, east = [], [], [], []
    
    for i in range(size):
        north.append(get_node_num(size, (0, i)))
        south.append(get_node_num(size, (size - 1, i)))
        west.append(get_node_num(size, (i, 0)))
        east.append(get_node_num(size, (i, size - 1)))

    return {
        'north': north, 
        'south': south, 
        'west': west,
        'east': east
    }

def get_node_num(size, space):
    return space[0] * size + space[1]
