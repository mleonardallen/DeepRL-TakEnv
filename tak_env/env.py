"""
Tak - A Beautiful Game
"""
import gym
import numpy as np
from tak_env.space import ActionSpace
import tak_env.board as Board
import tak_env.road as Road
import tak_env.score as Score
from tak_env.types import Player, Stone
from tak_env.viewer import Viewer
import copy

class TakEnv(gym.Env):
    """ TAK environment loop """

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 50
    }

    def __init__(self, board_size, scoring, pieces, capstones):

        """
        Args:
            board_size: size of the TAK board
            pieces: number of regular pieces per player
            capstones: number of capstones per player
        """
        assert isinstance(board_size, int) and board_size >= 3 and board_size <= 8, 'Invalid board size: {}'.format(board_size)
        assert isinstance(pieces, int) and pieces >= 10 and pieces <= 50, 'Invalid number of pieces: {}'.format(pieces)
        assert isinstance(capstones, int) and capstones >= 0 and capstones <= 2, 'Invalid number of capstones: {}'.format(capstones)

        # set board properties
        self.board_size = board_size
        self.capstones = capstones
        self.pieces = pieces

        self.action_space = ActionSpace(env=self)
        self.scoring = scoring
        self.viewer = Viewer(env=self)

        self.reset()

    def seed(self):
        pass

    def reset(self):

        self.done = False
        self.turn = np.random.choice([1, -1])
        self.reward = 0
        
        self.available_pieces = {}
        self.available_pieces[Player.WHITE.value] = {'pieces': self.pieces, 'capstones': self.capstones}
        self.available_pieces[Player.BLACK.value] = {'pieces': self.pieces, 'capstones': self.capstones}

        self.state = np.zeros((1, self.board_size, self.board_size))

        # multipart moves keep track of previous action
        self.continued_action = None
        return self.state

    def step(self, action):

        if self.done:
            return self.__feedback(action, reward=0, done=True)

        if action.get('action') == 'place':
            self.state = self.action_space.place(self.state, action, self.available_pieces, self.turn)
            self.update_pieces(self.available_pieces, action, self.turn)
        elif action.get('action') == 'move':
            self.state = self.action_space.move(self.state, action)

        # game ends when road is connected
        spaces = Board.get_owned_spaces(self.state, self.turn, [Stone.FLAT, Stone.CAPITAL])
        
        if Road.is_connected(self.board_size, spaces):
            score = self.__get_score(self.turn)
            return self.__feedback(action, reward=score, done=True)

        # game ends if no open spaces or if any player runs out of pieces
        if (
            (not Board.has_open_spaces(self.state)) or
            (len(self.get_available_piece_types(self.available_pieces, self.turn)) == 0)
        ):
            winner = Score.get_flat_winner(self.state)
            score = self.__get_score(winner)
            return self.__feedback(action, reward=score, done=True)

        # update player turn
        if action.get('terminal'):
            self.turn *= -1
            self.continued_action = None
        else:
            self.continued_action = action

        # game still going
        return self.__feedback(action, reward=0, done=False)

    def update_pieces(self, available, action, player):
        available = available.get(player)
        piece = action.get('piece')
        if piece is Stone.CAPITAL:
            available['capstones'] -= 1
        else:
            available['pieces'] -= 1
        return available

    def render(self, mode='human', close=False):
        if close:
            return
        self.viewer.render(self.state)

    def get_available_piece_types(self, available_pieces, player):
        """ Returns all available types of pieces to place """
        num_available = available_pieces.get(player)
        available = []
        if num_available.get('pieces', 0):
            available.append(Stone.FLAT)
        if num_available.get('pieces', 0):
            available.append(Stone.STANDING)
        if num_available.get('capstones', 0):
            available.append(Stone.CAPITAL)
        return available

    def __feedback(self, action, reward, done):

        self.done = done
        self.reward = reward

        return self.state, reward, self.done

    def __get_score(self, winner):
        if self.scoring == 'wins':
            return 1 if self.turn == winner else -1

        return Score.get_points(self.board_size, self.available_pieces, self.turn, winner)
