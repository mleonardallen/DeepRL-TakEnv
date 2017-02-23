from pydispatch import dispatcher
import argparse
import numpy as np
import copy

import gym
import env.tak
from env.board import Board
from agent.agent import RandomAgent, NFQAgent

def main(args):

    env = gym.make(args.env_id)

    agent_white = NFQAgent(env=env, symbol=Board.WHITE, learn=args.learn)
    agent_black = RandomAgent(env=env, symbol=Board.BLACK)

    rewards = []
    wins = 0
    losses = 0

    for i in range(int(args.iter)):

        state = env.reset()
        reward = 0
        env.turn = np.random.choice([1, -1])

        while True:

            state = copy.copy(state)
            # Get current active player
            agent = agent_white if env.turn == agent_white.symbol else agent_black
            # Take an action
            action = agent.act(state)
            player = env.turn
            state_prime, reward_prime, done, _ = env.step(action)
            player_prime = env.turn

            # allow agents to save experience
            experience = np.array([copy.copy(state), copy.copy(action), reward, copy.copy(state_prime), player, player_prime])
            dispatcher.send( signal='main.experience', sender={}, experience=experience)

            # update the state for next iteration
            state = state_prime
            reward = reward_prime

            if args.render:
                env.render()

            if done:
                break

        # record the "absorbing" state
        experience = np.array([copy.copy(state), None, reward, None, player, player_prime])
        dispatcher.send( signal='main.experience', sender={}, experience=experience)

        reward = reward * player
        rewards.append(reward)
        if reward > 0:
            wins+=1
        elif reward < 0:
            losses+=1

        if i % 100 == 0:
            print('episode', i)

    print('mean', np.mean(rewards), 'wins', wins, 'losses', losses)
    dispatcher.send(signal='main.complete', sender={})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('env_id', nargs='?', default='Tak3x3-wins-v0')

    parser.add_argument('--iter', dest='iter', default=1000)
    parser.add_argument('--render', dest='render', action='store_true')
    parser.add_argument('--learn', dest='learn', action='store_true')
    parser.add_argument('--log', dest='log', action='store_true')

    args = parser.parse_args()
    main(args)
