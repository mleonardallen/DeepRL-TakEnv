import numpy as np
import copy

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

    def __init__(self, value_function, env, **userconfig):

        self.env = env
        self.action_space = env.action_space
        self.symbol = userconfig.get('symbol')
        self.value_function = value_function

        self.config = {
            "epsilon": 0.1, # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "n_iter": 10000} # Number of iterations
        self.config.update(userconfig)

    def get_state_prime(self, action):
        action = copy.copy(action)
        action['hallucinate'] = True
        ob, reward, done, _ = self.env.step(action)
        return ob

    def act(self, observation, reward, done):

        # exploration
        epsilon = self.config["epsilon"]
        explore = np.random.choice([False, True], p = [1-epsilon, epsilon])
        valid_actions = self.action_space.get_valid_moves()
        if explore:
            return np.random.choice(valid_actions)

        # get best action, random if more than one best
        state_primes = np.array([self.get_state_prime(action) for action in valid_actions])
        values = self.value_function.predict(state_primes)
        actions = [action for idx, action in enumerate(valid_actions) if values[idx] == np.max(values)]
        action = np.random.choice(actions)
        return action
