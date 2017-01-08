import argparse
import gym
import env.tak
from env.board import Board
from agent.agent import RandomAgent, NFQAgent
from agent.value import CnnValueFunction

import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('env_id', nargs='?', default='Tak6x6-points-v0')
    args = parser.parse_args()
    env = gym.make(args.env_id)
    cnn = CnnValueFunction()

    agent_white = NFQAgent(value_function=cnn, env=env, symbol=Board.WHITE)
    agent_black = RandomAgent(env=env, symbol=Board.BLACK)

    episode_count = 1
    reward = 0
    done = False

    records = []

    for i in range(episode_count):
        print(i)
        state = env.reset()
        while True:
            agent = agent_white if env.turn == agent_white.symbol else agent_black
            action = agent.act(state, reward, done)
            state_prime, reward, done, _ = env.step(action)
            experience = np.array([state, state_prime, reward])
            records.append(experience)
            state = state_prime

            if done:
                break
