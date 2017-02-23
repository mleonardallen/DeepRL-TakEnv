import numpy as np
import copy
from pydispatch import dispatcher
import json
import os.path

from keras.layers.core import Dense, Flatten, SpatialDropout2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential

import time

from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
# from keras.layers.normalization import BatchNormalization
# from keras.callbacks import EarlyStopping

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
        self.learn = userconfig.get('learn')

        self.id = self.env.spec.id
        self.model = self.create_model();

        self.config = {
            "epsilon": 0.1, #Epsilon in epsilon greedy policies
            "n_iter": 30,
            "discount": 0.9,
            "nb_epoch": 3,
            "batch_size": 128,
        }
        self.config.update(userconfig)

        self.experiences = []

        if self.learn:
            dispatcher.connect(self.add_experience, signal='main.experience', sender=dispatcher.Any)
            dispatcher.connect(self.experience_replay, signal='main.complete', sender=dispatcher.Any)

    def create_model(self):
        input_shape=(self.env.board.size, self.env.board.size, self.env.board.height)
        model = Sequential([
            # Flatten(input_shape=input_shape),
            Convolution2D(2, 2, 2, init='normal', border_mode='same', activation='elu', input_shape=input_shape),
            SpatialDropout2D(0.3),

            Convolution2D(4, 2, 2, init='normal', border_mode='same', activation='elu'),
            SpatialDropout2D(0.3),

            Convolution2D(8, 2, 2, init='normal', border_mode='same', activation='elu'),
            SpatialDropout2D(0.3),

            Flatten(),
            Dense(1)
        ])
        model.summary()

        if os.path.isfile(self.id + '.h5'):
            model.load_weights(self.id + '.h5')

        model.compile(loss='mse', optimizer='adam')
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
        state_primes = np.array([self.get_state_prime(action) for action in valid_actions])
        values = self.model.predict(state_primes)
        actions = [action for idx, action in enumerate(valid_actions) if values[idx] == np.max(values)]
        action = np.random.choice(actions)
        return action

    def add_experience(self, experience):
        """ add experience for experience replay """
        self.experiences.append(experience)

    def experience_replay(self):
        experiences = np.array(self.experiences)

        n_iter = self.config["n_iter"]
        for j in range(n_iter):
            print("\nIteration:", j)

            # 1) Get values for next states
            state_primes = []
            for idx, experience in enumerate(experiences):
                state, action, reward, state_prime, player, player_prime = experience
                # prediction from next players perspective
                if state_prime is not None:
                    state_primes.append(state_prime * player_prime + 0)
            q_primes = self.model.predict(np.array(state_primes))

            # 2) Update values according to bellman equation
            X, y = [], []
            for idx, experience in enumerate(experiences):
                state, action, reward, state_prime, player, player_prime = experience
                q_prime = q_primes[idx][0] if idx < len(q_primes) else 0

                # Q(s,a) <- r + γ * max_a' Q(s',a')
                reward = reward * player
                future = q_prime * player_prime
                γ = self.config["discount"]

                X.append(state)
                y.append(reward + γ * future)
            X, y = np.array(X), np.array(y)

            # 3) Fit
            self.model.fit(
                X, y,
                batch_size=self.config["batch_size"],
                nb_epoch=self.config["nb_epoch"],
                shuffle=True
            )

        self.save()

    def save(self):
        with open(self.id + '.json' , 'w') as outfile:
            json.dump(self.model.to_json(), outfile)
        self.model.save_weights(self.id + '.h5')
