"""
Tak - A Beautiful Game
"""

import gym
import numpy as np

from env.graph import ConnectionGraph
from env.space import ActionSpace
from env.board import Board

class TakEnv(gym.Env):
    """ TAK environment loop """

    def __init__(self, board_size=4):
        """
        Args:
            board_size: size of the TAK board
        """
        assert isinstance(board_size, int) and board_size >= 3, 'Invalid board size: {}'.format(board_size)
        Board.size = board_size

        self.action_space = ActionSpace(env=self)
        self.graph = ConnectionGraph()

        self._reset()

    def _reset(self):
        self.done = False
        self.turn = 1

        # multipart moves keep track of previous action
        self.continued_action = None

        # reset board state
        Board.reset()
        self.graph.reset()

        return self._state()

    def _state(self):
        return {
            'board': Board.state,
            'turn': self.turn,
            'black': Board.get_available_pieces(Board.BLACK),
            'white': Board.get_available_pieces(Board.WHITE)
        }

    def _step(self, action):

        if self.done:
            return self._feedback(0)

        # player takes an action
        if action.get('action') == 'place':
            Board.place(action, self.turn)
        elif action.get('action') == 'move':
            Board.move(action)

        # game ends if win
        if self.is_win(self.turn):
            self.done = True
            return self._feedback(1)

        # game ends if no open spaces
        if not Board.has_open_spaces():
            self.done = True
            return self._feedback(0)

        # game ends if any player runs out of pieces
        if len(Board.get_available_piece_types(self.turn)) == 0:
            self.done = True
            return self._feedback(0)

        # update player turn
        if action.get('terminal'):
            self.turn *= -1
            self.continued_action = None
        else:
            self.continued_action = action


        return self._feedback(0)

    def _feedback(self, reward):
        return self._state(), reward, self.done, {}

    def _render(self, mode='human', close=False):
        if close:
            return

    def is_win(self, player):
        return self.graph.is_connected(player)