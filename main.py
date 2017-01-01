import argparse
import logging
import os
import sys
import gym
import env.tak
from agent import RandomAgent
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='Tak-v0')
    # parser.add_argument('board_size', nargs='?', default=3)
    args = parser.parse_args()
    env = gym.make(args.env_id)
    agent = RandomAgent(env)

    df = pd.DataFrame()

    episode_count = 1
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        # while True:
        for i in range(10):
            action = agent.act(ob, reward, done)
            
            ob, reward, done, _ = env.step(action)

            if done:
                break

            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
