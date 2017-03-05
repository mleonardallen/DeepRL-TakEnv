import numpy as np
import copy
import json
import os.path

from keras.layers.core import Dense, Flatten, SpatialDropout2D, Dropout
from keras.layers import Input, Convolution2D, merge
from keras.models import Sequential, Model

from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.regularizers import l2

from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Agent(object):

    def __init__(self, **userconfig):

        self.env = userconfig.get('env')
        self.symbol = userconfig.get('symbol')

class RandomAgent(Agent):

    def act(self, state):
        return self.env.action_space.sample()

class LearnerAgent(Agent):

    """
    Learner Agent
    """

    def __init__(self, **userconfig):

        Agent.__init__(self, **userconfig)

        self.config = {
            # exploration probability
            "epsilon": 0.1
        }
        self.config.update(userconfig)

        self.value_function = ValueFunction(
            size=self.env.board.size,
            height=self.env.board.height,
            id=self.env.spec.id
        )

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

        state = copy.copy(result[0])
        return state * self.symbol + 0

    def act(self, state):

        # exploration
        ɛ = self.config["epsilon"]
        explore = np.random.choice([False, True], p = [1-ɛ, ɛ])

        valid_actions = self.env.action_space.get_valid_moves()
        if explore:
            return self.env.action_space.sample()

        # get best action, random if more than one best
        state_primes = np.array([self.get_state_prime(action) for action in valid_actions])
        values = self.value_function.get_value(state_primes)
        actions = [action for idx, action in enumerate(valid_actions) if values[idx] == np.max(values)]
        action = np.random.choice(actions)
        return action

class ValueFunction():

    def __init__(self, **userconfig):

        self.config = {
            "discount": 0.95,
            "size": 0,
            "height": 0,
            "id": ""
        }
        self.config.update(userconfig)

        self.size = self.config.get('size')
        self.height = self.config.get('height')
        self.id = self.config.get('id')

        self.model = self.create_model();

    def get_value(self, state):
        return self.model.predict(state)

    def create_model(self):

        height = self.height
        input_shape=(self.size, self.size, height)
        inputs = Input(shape=input_shape)

        layer = BatchNormalization()(inputs)
        layer = self.Fire_module(layer, [height, height*2, height*2, height*2])
        layer = SpatialDropout2D(0.3)(layer)

        layer = BatchNormalization()(layer)
        layer = self.Fire_module(layer, [height*2, height*3, height*3, height*3])
        layer = SpatialDropout2D(0.3)(layer)

        layer = Flatten()(layer)
        layer = Dense(100, activation='elu')(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(50, activation='elu')(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(1)(layer)

        model = Model(input=inputs,output=layer)

        model.compile(loss='mse', optimizer='adam')
        model.summary()

        if os.path.isfile(self.id + '.h5'):
            model.load_weights(self.id + '.h5')

        return model

    def Fire_module(self, input_layer,nb_kernel):
        squeeze  = Convolution2D(nb_kernel[0], 1, 1, border_mode='same', activation='elu')(input_layer)
        expand_1x1 = Convolution2D(nb_kernel[1], 1, 1, border_mode='same', activation='elu')(squeeze)
        expand_2x2 = Convolution2D(nb_kernel[2], 2, 2, border_mode='same', activation='elu')(squeeze)
        expend_3x3 = Convolution2D(nb_kernel[3], 3, 3, border_mode='same', activation='elu')(squeeze)
        return merge([expand_1x1,expand_2x2, expend_3x3], mode='concat', concat_axis=1)

    def experience_replay(self, experiences, n_iter=1, batch_size=1024, nb_epoch=1):

        for j in range(n_iter):
            print("\nIteration:", j + 1)

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
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                shuffle=True
            )

        self.save()

    def save(self):
        # with open(self.id + '.json' , 'w') as outfile:
        #     json.dump(self.model.to_json(), outfile)
        self.model.save_weights(self.id + '.h5')
