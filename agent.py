import numpy as np

import numpy as np
import pandas as pd
import json

import os.path
import copy

from keras.preprocessing.image import ImageDataGenerator, Iterator, flip_axis
from keras.layers.core import Dense, Flatten, Activation, SpatialDropout2D, Lambda, Dropout
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from keras.objectives import mse
from sklearn.model_selection import train_test_split

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, **userconfig):
        self.env = userconfig.get('env')
        self.symbol = userconfig.get('symbol')

    def act(self, observation, reward, done):
        return self.env.action_space.sample()

class NFQAgent(object):

    """
    Agent implementing Neural Fitted Q-learning.
    """

    def __init__(self, env, **userconfig):

        self.env = env
        self.action_space = env.action_space
        self.symbol = userconfig.get('symbol')

        self.config = {
            "learning_rate" : 0.1,
            "epsilon": 0.05, # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "n_iter": 10000} # Number of iterations
        self.config.update(userconfig)
        self.model = self.create_model();

    def create_model(self):
        model = Sequential([
            # Normalize to keep weight values small with zero mean, improving numerical stability.
            BatchNormalization(axis=1, input_shape=(5, 6, 6)),
            # Conv 2x2
            Convolution2D(24, 2, 2, border_mode = 'same', activation = 'elu'),
            SpatialDropout2D(0.2),
            # Conv 2x2
            Convolution2D(36, 2, 2, border_mode = 'same', activation = 'elu'),
            SpatialDropout2D(0.2),
            # Flatten
            Flatten(),
            # Fully Connected
            Dense(100, activation = 'elu', W_regularizer = l2(1e-6)),
            Dense(50, activation = 'elu', W_regularizer = l2(1e-6)),
            Dense(10, activation = 'elu', W_regularizer = l2(1e-6)),
            Dense(1)
        ])
        model.summary()

        if os.path.isfile('model.h5'):
            model.load_weights('model.h5')

        model.compile(loss = NFQAgent.loss, optimizer = 'adam')
        return model

    @staticmethod
    def loss(y_true, y_pred):
        return mse(y_true, y_pred)

    def get_qsa(self, action):
        """ """
        # see what the state looks like after taking the action
        return 1
        action = copy.copy(action)
        action['hallucinate'] = True
        ob, reward, done, _ = self.env.step(action)
        state = ob.get('board')
        return float(self.model.predict(np.array([state]), batch_size = 1))

    def act(self, observation, reward, done):

        # exploration
        epsilon = self.config["epsilon"]
        explore = np.random.choice([False, True], p = [1-epsilon, epsilon])
        valid_actions = self.action_space.get_valid_moves()
        if explore:
            return np.random.choice(valid_actions)

        # get best action, random if more than one best
        values = [self.get_qsa(action) for action in valid_actions]
        actions = [action for idx, action in enumerate(valid_actions) if values[idx] == np.max(values)]
        action = np.random.choice(actions)
        return action
