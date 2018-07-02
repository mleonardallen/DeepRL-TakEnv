import numpy as np
import copy
import json
from tak.value import ValueFunction

class Agent(object):

    def __init__(self, **userconfig):

        self.env = userconfig.get('env')
        self.symbol = userconfig.get('symbol')
        self.value_function = userconfig.get('value_function')

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
        epsilon = self.config["epsilon"]
        explore = np.random.choice([False, True], p = [1-epsilon, epsilon])

        valid_actions = self.env.action_space.get_valid_moves()
        if explore:
            return self.env.action_space.sample()

        # get best action, random if more than one best
        state_primes = np.array([self.get_state_prime(action) for action in valid_actions])
        values = self.value_function.get_value(state_primes)
        actions = [action for idx, action in enumerate(valid_actions) if values[idx] == np.max(values)]
        action = np.random.choice(actions)
        return action
