"""
Tak - A Beautiful Game
"""

from six import StringIO
import sys
import gym
from gym import spaces
import numpy as np
from gym import error
from gym.utils import seeding
from env.graph import TakGraph

import math

class TakEnv(gym.Env):
    """
    Tak environment
    """
    BLACK = -1
    WHITE = 1

    CONFIG = {
        3: {'pieces': 10, 'capstones': 0},
        4: {'pieces': 15, 'capstones': 0},
        5: {'pieces': 21, 'capstones': 1},
        6: {'pieces': 30, 'capstones': 1},
        7: {'pieces': 40, 'capstones': -1},
        8: {'pieces': 50, 'capstones': 2},
    }

    MAX_HEIGHT = 3

    def __init__(self, board_size=4):
        """
        Args:
            board_size: size of the Tak board
        """
        assert isinstance(board_size, int) and board_size >= 3, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size
        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.config = TakEnv.CONFIG.get(board_size)
        self.graph = TakGraph(board_size)
        self._reset()

    def _reset(self):
        self.state = np.zeros((TakEnv.MAX_HEIGHT, self.board_size, self.board_size))
        self.done = False
        self.turn = 1

        return self.state

    def _step(self, action):

        print('action', action)

        if self.done:
            return self._feedback(0)

        if not self.valid_move(action):
            self.done = True
            return self._feedback(-1)

        try:
            self.place(action)
            self.turn *= -1
        except Exception as e:
            return self._feedback(-1)

        if self.is_win():
            self.done = True
            return self._feedback(1)

        return self._feedback(0)

    def _feedback(self, reward):
        return self.state, reward, self.done, {'state': self.state}

    def _render(self, mode='human', close=False):
        if close:
            return

    def valid_move(self, action):
        space = self.get_space_from_action(action)
        board = self.get_top_layer()
        return board[space] == 0

    def place(self, action):
        space = self.get_space_from_action(action)
        available = self.get_first_available_layer(space)

        self.state[available][space] = self.turn

    def get_space_from_action(self, num):
        row = math.floor((num - 1) / self.board_size)
        col = (num - 1) % self.board_size
        return row, col

    def get_top_layer(self):
        merged = []
        for layer in self.state:
            if len(merged):
                layer[merged !=0] = merged[merged !=0]
            merged = layer
        return merged

    def get_first_available_layer(self, space):
        reverse_state = self.state[::-1]
        for idx in reversed(range(TakEnv.MAX_HEIGHT)):
            if self.state[idx][space] == 0:
                return idx

        raise RuntimeError('Exceeded MAX_HEIGHT')

    def is_win(self):
        board = self.get_top_layer()
        return self.graph.is_connected(board, self.turn)