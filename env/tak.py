"""
Tak - A Beautiful Game
"""

import gym
import numpy as np

from env.graph import ConnectionGraph
from env.space import ActionSpace
from env.board import Board

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
        7: {'pieces': 40, 'capstones': 2},
        8: {'pieces': 50, 'capstones': 2},
    }

    MAX_HEIGHT = 3

    def __init__(self, board_size=4):
        """
        Args:
            board_size: size of the Tak board
        """
        assert isinstance(board_size, int) and board_size >= 3, 'Invalid board size: {}'.format(board_size)
        Board.size = board_size

        self.action_space = ActionSpace(env=self)
        self.graph = ConnectionGraph()

        self._reset()

    def _reset(self):
        self.done = False
        self.turn = 1

        # keep track of available pieces
        config = TakEnv.CONFIG.get(Board.size)
        self.available_pieces = {TakEnv.BLACK: config, TakEnv.WHITE: config}

        Board.reset()
        self.graph.reset()

        return self._state()

    def _state(self):
        return {
            'board': Board.state
        }

    def _step(self, action):

        if self.done:
            print(Board.state)
            return self._feedback(0)

        # player takes an action
        if action.get('action') == 'place':
            Board.place(action, self.turn)
        elif action.get('action') == 'move':
            Board.move(action)

        # game ends if win
        if self.is_win():
            self.done = True
            return self._feedback(1)

        # game ends if no open spaces
        if not Board.has_open_spaces():
            self.done = True
            return self._feedback(0)

        # update player turn
        if action.get('terminal', False):
            self.turn *= -1

        return self._feedback(0)

    def _feedback(self, reward):
        return self._state(), reward, self.done, {}

    def _render(self, mode='human', close=False):
        if close:
            return

    def get_available(self):
        return self.available_pieces.get(self.turn)

    def is_win(self):
        return self.graph.is_connected(self.turn)