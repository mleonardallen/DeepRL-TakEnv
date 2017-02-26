import numpy as np
import copy
import json
import os.path

from keras.layers.core import Dense, Flatten, SpatialDropout2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential

import time

from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
# from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

class RandomAgent(object):
    def __init__(self, **userconfig):
        self.env = userconfig.get('env')
        self.symbol = userconfig.get('symbol')

    def act(self, state):
        return self.env.action_space.sample()

class NFQAgent(object):

    """
    Agent implementing Neural Fitted Q-learning.
    """

    def __init__(self, **userconfig):

        self.env = userconfig.get('env')
        self.symbol = userconfig.get('symbol')
        self.id = self.env.spec.id
        self.model = self.create_model();

        self.config = {
            "epsilon": 0.1, #Epsilon in epsilon greedy policies
            "n_iter": 1,
            "discount": 0.9,
            "nb_epoch": 3,
            "batch_size": 128,
        }
        self.config.update(userconfig)

    def create_model(self):
        height = self.env.board.height
        input_shape=(self.env.board.size, self.env.board.size, height)
        model = Sequential([

            Convolution2D(height*2, 2, 2, init='normal', border_mode='same', activation='elu', input_shape=input_shape),
            SpatialDropout2D(0.2),

            Convolution2D(height*3, 2, 2, init='normal', border_mode='same', activation='elu'),
            SpatialDropout2D(0.2),

            Convolution2D(height*4, 2, 2, init='normal', border_mode='same', activation='elu'),
            SpatialDropout2D(0.2),

            Flatten(),

            Dense(1)
        ])
        model.summary()

        if os.path.isfile(self.id + '.hdf5'):
            model.load_weights(self.id + '.hdf5')

        model.compile(loss='mse', optimizer='adam', callbacks=[
            EarlyStopping(monitor='val_loss'),
            ModelCheckpoint(filepath=self.id + '.hdf5', monitor='val_loss', save_best_only=True)
        ])
        return model

    def get_state_prime(self, action):
        # copy so we can restore after hallucination
        # todo encapsulate in __copy__ method of the env
        turn = self.env.turn
        reward = self.env.reward
        board = copy.copy(self.env.board)
        continued_action = copy.copy(self.env.continued_action)

        # hallucinate move
        result = self.env.step(action)

        # restore environment
        self.env.done = False
        self.env.turn = turn
        self.env.reward = reward
        self.env.board = board
        self.env.continued_action = continued_action

        return copy.copy(result[0])

    def act(self, state):

        # exploration
        ɛ = self.config["epsilon"]
        explore = np.random.choice([False, True], p = [1-ɛ, ɛ])

        valid_actions = self.env.action_space.get_valid_moves()
        if explore:
            return self.env.action_space.sample()

        # get best action, random if more than one best
        state_primes = np.array([self.get_state_prime(action) for action in valid_actions]) * self.symbol + 0
        values = self.model.predict(state_primes)
        actions = [action for idx, action in enumerate(valid_actions) if values[idx] == np.max(values)]
        action = np.random.choice(actions)
        return action

    def experience_replay(self, experiences, n_iter=0):

        for j in range(n_iter):
            print("\nIteration:", j)

            ###
            # 1) Get values for next states
            ###

            booleanMask = experiences['state_prime'].notnull()

            # get state prime, and keep track of indexes for later when mapping back onto full array
            state_primes = experiences[booleanMask]['state_prime']
            idxs = state_primes.index.values
            state_primes = np.array([state for state in state_primes.values])

            # see state prime from player prime's perspective and predict value
            player_primes = experiences[booleanMask]['player_prime']
            state_primes = np.multiply(state_primes, player_primes.values[:, np.newaxis, np.newaxis, np.newaxis]) + 0
            pred_q_primes = self.model.predict(state_primes).reshape(-1)

            # map predictions back onto full array containing absorbing states
            q_primes = np.zeros(len(experiences.index))
            for j, idx in enumerate(idxs):
                q_primes[idx] = pred_q_primes[j]
            # minimax - good reward for player prime is bad reward for player
            player_primes = experiences['player_prime'].values
            q_primes = np.multiply(q_primes, player_primes)

            ###
            # 2) Update values according to bellman equation
            ###

            X = experiences['state'].values
            X = np.array([x for x in X])
            rewards = experiences['reward'].values
            players = experiences['player'].values
            # minimax - good reward for player prime is bad reward for player
            rewards = np.multiply(rewards, players)

            # Q(s,a) <- r + γ * max_a' Q(s',a')
            γ = self.config["discount"]
            y = np.add(rewards, γ * q_primes)

            ###
            # 3) Fit
            ###
            self.model.fit(
                X, y,
                batch_size=self.config["batch_size"],
                nb_epoch=self.config["nb_epoch"],
                shuffle=True
            )
