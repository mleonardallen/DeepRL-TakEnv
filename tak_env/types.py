from enum import Enum

class Stone(Enum):
    """ Types of TAK game pieces (stones) """
    FLAT = 1
    STANDING = 2
    CAPITAL = 3
    EMPTY = 0

class StoneLetter(Enum):
    FLAT = 'F'
    STANDING = 'S'
    CAPITAL = 'C'

class Player(Enum):
    BLACK = -1
    WHITE = 1

class Direction(Enum):
    UP = '+'
    DOWN = '-'
    LEFT = '<'
    RIGHT = '>'