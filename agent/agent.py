import numpy as np
import copy
from pydispatch import dispatcher
import json
import os.path

from keras.layers.core import Dense, Flatten, SpatialDropout2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential

# from keras.layers.pooling import MaxPooling2D
# from keras.regularizers import l2
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

        self.id = self.env.spec.id
        self.model = self.create_model();

        self.config = {
            "epsilon": 0.1, #Epsilon in epsilon greedy policies
            "alpha": 0.9,
            "n_iter": 100,
            "discount": 0.8,
            "nb_epoch": 3,
            "batch_size": 256
        }
        self.config.update(userconfig)

        self.experiences = []
        dispatcher.connect(self.add_experience, signal='main.experience', sender=dispatcher.Any)
        dispatcher.connect(self.learn, signal='main.complete', sender=dispatcher.Any)

    def get_state_prime(self, action):
        # copy action so we don't modifiy original action
        action = copy.copy(action)
        # save turn so we can restore after hallucination
        turn = self.env.turn
        action['hallucinate'] = True
        ob, reward, done, _ = self.env.step(action)
        # restore turn
        self.env.turn = turn
        return ob

    def act(self, state):

        # exploration
        ɛ = self.config["epsilon"]
        explore = np.random.choice([False, True], p = [1-ɛ, ɛ])

        valid_actions = self.env.action_space.get_valid_moves()
        if explore:
            return self.env.action_space.sample()

        # get best action, random if more than one best
        state_primes = np.array([self.get_state_prime(action) for action in valid_actions])
        values = self.predict(state_primes)
        actions = [action for idx, action in enumerate(valid_actions) if values[idx] == np.max(values)]
        action = np.random.choice(actions)
        return action

    def add_experience(self, experience):
        """ add experience for experience replay """
        self.experiences.append(experience)

    def learn(self):
        experiences = np.array(self.experiences)

        n_iter = self.config["n_iter"]
        for j in range(n_iter):
            print("Iteration:", j)

            # 1) Get values for next states
            state_primes = []
            states = []
            for idx, experience in enumerate(experiences):
                state, action, reward, state_prime, player, player_prime = experience

                # prediction from next players perspective
                states.append(state * player + 0)
                state_primes.append(state_prime * player_prime + 0)

            qs = self.predict(np.array(states))
            q_primes = self.predict(np.array(state_primes))

            # 2) Update values according to bellman equation
            X, y = [], []
            for idx, q_prime in enumerate(q_primes):
                state, action, reward, state_prime, player, player_prime = experiences[idx]

                # Q(s,a) <- r + γ * max_a' Q(s',a')
                reward = reward * player
                future = q_prime[0] * player_prime

                print('---')
                print('reward', reward, 'player', player)
                print('future', future, 'player_prime', player_prime)
                print(action)
                # self.env.viewer.render(state)
                self.env.viewer.render(state_prime)

                γ = self.config["discount"]
                α = self.config["alpha"]

                value = qs[idx][0]
                value = (1 - α) * value + α * (reward + γ * future)

                X.append(state)
                y.append(value)

            X, y = np.array(X), np.array(y)

            # 3) Fit
            self.fit(X, y)

        self.save()


    def create_model(self):
        input_shape=(self.env.board.size, self.env.board.size, self.env.board.height)
        model = Sequential([
            Flatten(input_shape=input_shape),
            Dense(100, activation='elu'),
            Dropout(0.2),
            Dense(20, activation='elu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.summary()

        if os.path.isfile(self.id + '.h5'):
            model.load_weights(self.id + '.h5')

        model.compile(loss='mse', optimizer='adam')
        return model

    def fit(self, X, y):
        # earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
        self.model.fit(X, y, batch_size=self.config["batch_size"], nb_epoch=self.config["nb_epoch"], shuffle=True)

    def predict(self, X):
        return self.model.predict(X)

    def save(self):
        with open(self.id + '.json' , 'w') as outfile:
            json.dump(self.model.to_json(), outfile)
        self.model.save_weights(self.id + '.h5')
