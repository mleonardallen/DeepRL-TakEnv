import argparse
import gym
import env.tak
from env.board import Board
from agent import RandomAgent

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('env_id', nargs='?', default='Tak4x4-v0')
    args = parser.parse_args()
    env = gym.make(args.env_id)

    agent_white = RandomAgent(env=env, symbol=Board.WHITE)
    agent_black = RandomAgent(env=env, symbol=Board.BLACK)

    episode_count = 1
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            agent = agent_white if env.turn == agent_white.symbol else agent_black
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)

            if done:
                break

        print('reward', reward, 'player', env.turn)