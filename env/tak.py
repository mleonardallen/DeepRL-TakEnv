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

    def __init__(self, board_size=4, scoring='points'):
        """
        Args:
            board_size: size of the TAK board
        """
        assert isinstance(board_size, int) and board_size >= 3, 'Invalid board size: {}'.format(board_size)
        Board.size = board_size

        self.action_space = ActionSpace(env=self)
        self.graph = ConnectionGraph()
        self.scoring = scoring

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
            score = self.get_score(self.turn)
            return self._feedback(score)

        # game ends if no open spaces or if any player runs out of pieces
        if (
            (not Board.has_open_spaces()) or
            (len(Board.get_available_piece_types(self.turn)) == 0)
        ):
            self.done = True
            winner = Board.get_flat_winner()
            score = self.get_score(winner)
            return self._feedback(score)

        # update player turn
        if action.get('terminal'):
            self.turn *= -1
            self.continued_action = None
        else:
            self.continued_action = action

        # game still going
        return self._feedback(0)

    def _feedback(self, reward):
        return self._state(), reward, self.done, {}

    def _render(self, mode='human', close=False):
        if close:
            return

    def get_score(self, winner):
        if self.scoring == 'win':
            return 1 if self.turn == winner else -1
        return Board.get_points(self.turn, winner)

    def is_win(self, player):
        return self.graph.is_connected(player)