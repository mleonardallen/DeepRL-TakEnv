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

    parser.add_argument('env_id', nargs='?', default='Tak3x3-wins-v0')
    args = parser.parse_args()
    env = gym.make(args.env_id)
    cnn = CnnValueFunction()

    agent_white = NFQAgent(value_function=cnn, env=env, symbol=Board.WHITE)
    agent_black = RandomAgent(env=env, symbol=Board.BLACK)

    episode_count = 1000
    reward = 0
    done = False

    # 1) Gather X_train
    records = []
    for i in range(episode_count):
        state = env.reset()
        while True:
            agent = agent_white if env.turn == agent_white.symbol else agent_black
            action = agent.act(state, reward, done)
            player = env.turn
            state_prime, reward, done, _ = env.step(action)
            # reward in terms of learning agent
            reward = (reward * player)

            player_prime = env.turn
            # also store the next player up for minimax/zero sum game
            experience = np.array([state, state_prime, reward, player_prime])

            records.append(experience)
            state = state_prime

            if done:
                break

        if i % 20 == 0:
            print('episode', i, 'reward', reward, env.turn)

    records = np.array(records)
    discount = 0.8
    print('average', np.mean(records, axis=0)[2])

    for j in range(30):

        # 2) X_train, y_train
        state_primes = []
        for i in records:
            state_prime = i[1]
            player_prime = i[3]
            # prediction must be made from next players perspective
            state_primes.append(state_prime * player_prime + 0)
        state_primes = np.array(state_primes)
        q_primes = cnn.predict(state_primes)

        print(q_primes)

        X, y = [], []
        for idx, q_prime in enumerate(q_primes):
            reward = records[idx][2]
            state = records[idx][0]
            player_prime = records[idx][3]
            future = q_prime[0] * player_prime

            X.append(state)
            y.append(reward + discount * future)

        X, y = np.array(X), np.array(y)

        # 3) Fit
        cnn.fit(X, y)
    cnn.save()

    print('average', np.mean(records, axis=0)[2])
# save weights for next run
