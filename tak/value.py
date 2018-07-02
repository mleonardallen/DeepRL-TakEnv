from keras.layers.core import Dense, Flatten, SpatialDropout2D, Dropout
from keras.layers import Input, Conv2D, concatenate
from keras.models import Sequential, Model

from keras.optimizers import SGD, Adam

from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.regularizers import l2

from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tak.generator import StateGenerator

import math
import time
import os.path
import numpy as np

class ValueFunction():

    def __init__(self, **userconfig):

        self.config = {
            "discount": 0.95,
            "size": 0,
            "height": 0,
            "id": "",
            "weight_decay": 0.05,
            "learning_rate": 0.00025
        }
        self.config.update(userconfig)

        self.size = self.config.get('size')
        self.height = self.config.get('height')
        self.id = self.config.get('id')
        self.weight_decay = self.config.get('weight_decay')
        self.learning_rate = self.config.get('learning_rate')

        self.model = self.create_model();

    def get_value(self, state):
        return self.model.predict(state)

    def create_model(self):

        height = self.height
        input_shape=(self.size, self.size, height)
        inputs = Input(shape=input_shape)

        layer = BatchNormalization()(inputs)
        layer = self.Fire_module(layer, [height, height, height, height])
        layer = SpatialDropout2D(0.3)(layer)

        layer = BatchNormalization()(layer)
        layer = self.Fire_module(layer, [height, height, height, height])
        layer = SpatialDropout2D(0.3)(layer)

        layer = BatchNormalization()(layer)
        layer = self.Fire_module(layer, [height, height, height, height])
        layer = SpatialDropout2D(0.3)(layer)

        layer = Flatten()(layer)
        layer = Dense(100, activation='elu', kernel_regularizer=l2(self.weight_decay))(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(50, activation='elu', kernel_regularizer=l2(self.weight_decay))(layer)
        layer = Dropout(0.3)(layer)

        layer = Dense(1)(layer)

        model = Model(inputs=inputs,outputs=layer)

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()

        if os.path.isfile(self.id + '.h5'):
            model.load_weights(self.id + '.h5')

        return model

    def Fire_module(self, input_layer,nb_kernel):
        squeeze  = Conv2D(nb_kernel[0], (1, 1), padding='same', activation='elu', kernel_regularizer=l2(self.weight_decay))(input_layer)
        expand_1x1 = Conv2D(nb_kernel[1], (1, 1), padding='same', activation='elu', kernel_regularizer=l2(self.weight_decay))(squeeze)
        expand_2x2 = Conv2D(nb_kernel[2], (2, 2), padding='same', activation='elu', kernel_regularizer=l2(self.weight_decay))(squeeze)
        # expend_3x3 = Convolution2D(nb_kernel[3], 3, 3, border_mode='same', activation='elu')(squeeze)
        return concatenate([expand_1x1,expand_2x2], axis=1)

    def experience_replay(self, experiences, n_iter=1, batch_size=64512, nb_epoch=1):

        for j in range(n_iter):
            print("\nIteration:", j + 1)

            ###
            # 1) Get values for next states
            ###

            # get state prime, and keep track of indexes for later when mapping back onto full array
            print("Get S' (Excluding Absorbing States)")
            start = time.time()
            booleanMask = experiences['state_prime'].notnull()
            state_primes = experiences[booleanMask]['state_prime']
            idxs = state_primes.index.values
            # reshape hack so we can change perspective later
            state_primes = np.array([state for state in state_primes.values])
            end = time.time()
            print("time:", end-start)

            # see state prime from player prime's perspective and predict value
            print("Convert S' to Player' Perspective")
            start = time.time()
            player_primes = experiences[booleanMask]['player_prime']
            state_primes = np.multiply(state_primes, player_primes.values[:, np.newaxis, np.newaxis, np.newaxis]) + 0
            end = time.time()
            print("time:", end-start)

            print("Predict Q' Values")
            start = time.time()
            pred_q_primes = self.model.predict(state_primes, batch_size=batch_size).reshape(-1)
            end = time.time()
            print("time:", end-start)

            # map predictions back onto full array containing absorbing states
            print("Q' Values (Including Absorbing States)")
            start = time.time()
            q_primes = np.zeros(len(experiences.index))
            np.put(q_primes, idxs, pred_q_primes)
            # minimax - good reward for player prime is bad reward for player
            player_primes = experiences['player_prime'].values
            q_primes = np.multiply(q_primes, player_primes)
            end = time.time()
            print("time:", end-start)

            ###
            # 2) Update values according to bellman equation
            ###

            print("Update Target Values According to Bellman Equation.")
            start = time.time()
            X_train = experiences['state'].values
            X_train = np.array([x for x in X_train])
            rewards = experiences['reward'].values
            players = experiences['player'].values
            # minimax - good reward for player prime is bad reward for player
            rewards = np.multiply(rewards, players)

            # Q(s,a) <- r + discount * max_a' Q(s',a')
            discount = self.config["discount"]
            y_train = np.add(rewards, discount * q_primes)

            end = time.time()
            print("time:", end-start)

            ###
            # 3) Fit
            ###

            print("3) Fit model.")
            datagen = StateGenerator()
            self.model.fit_generator(
                datagen.flow(X_train, y_train, batch_size=batch_size, flip_prob=0.5, rotate_prob=0.5),
                steps_per_epoch=math.floor(len(y_train)/batch_size),
                epochs=nb_epoch
            )

        self.save()

    def save(self):
        self.model.save_weights(self.id + '.h5')
