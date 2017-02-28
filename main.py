import argparse
import numpy as np
import pandas as pd
import copy
import json

import gym

import tak.env
from tak.board import Board
from tak.agent import RandomAgent, NFQAgent
from tak.experience import Experience
from peewee import fn

def main(args):

    mode = args.mode
    env = gym.make(args.env_id)

    agent_white = NFQAgent(env=env, symbol=Board.WHITE, epsilon=args.epsilon)

    if args.player == 'random':
        agent_black = RandomAgent(env=env, symbol=Board.BLACK)
    elif args.player == 'trained':
        agent_black = NFQAgent(env=env, symbol=Board.BLACK)
    elif args.player == 'human':
        # todo
        pass

    if mode == 'record':
        for i in range(int(args.iter)):

            if i % 20 == 0 and i > 0:
                print('Episode', i)

            loop_experiences = agent_environment_loop(env, agent_white, agent_black, render=False)
            q = Experience.insert_many(loop_experiences)
            q.execute()

    if mode == 'play':
        results = []
        for i in range(int(args.iter)):

            if i % 20 == 0 and i > 0:
                print('Episode', i)

            experiences = agent_environment_loop(env, agent_white, agent_black, render=args.render)
            state, action, reward, state_prime, player, player_prime, env_id = map(experiences[-1].get, 
                ('state', 'action', 'reward', 'state_prime', 'player', 'player_prime', 'env_id')
            )

            # convert reward to player one's perspective
            reward = reward * player
            results.append([reward, reward > 0, reward < 0, reward == 0])

        # Print Results
        sums = np.sum(results, axis=0)
        mean = np.mean(results, axis=0)

        total, wins, losses, ties = sums

        print('\nTotal Score:', total, 'Average Score:', mean[0])
        print('Wins:', wins, 'Losses:', losses, 'Ties:', ties)

    if mode == 'train':
        query = Experience.select() \
            .where(Experience.env_id == args.env_id) \
            .order_by(fn.Rand()) \
            .limit(args.limit)

        experiences = pd.DataFrame([x for x in query.dicts()])

        experiences.fillna(False, inplace=True)
        experiences.loc[:, 'state'] = experiences['state'].apply(convert)
        experiences.loc[:, 'state_prime'] = experiences['state_prime'].apply(convert)

        agent_white.experience_replay(experiences, n_iter=args.iter)

def convert(state):
    if state:
        return np.asarray(json.loads(state))
    return None

def agent_environment_loop(env, agent_white, agent_black, render=False):
    """ agent environment loop """

    experiences = []
    state = env.reset()
    reward = 0
    env.turn = np.random.choice([1, -1])

    while True:

        state = copy.copy(state)
        # Get current active player
        agent = agent_white if env.turn == agent_white.symbol else agent_black
        # Get action to take
        action = agent.act(state)
        # take action
        player = env.turn
        state_prime, reward_prime, done, _ = env.step(action)
        player_prime = env.turn

        # record the experience
        experiences.append({
            'state': json.dumps(state.tolist()),
            'action': copy.copy(action),
            'reward': reward,
            'state_prime': json.dumps(state_prime.tolist()),
            'player': player,
            'player_prime': player_prime,
            'env_id': env.spec.id
        })

        # update the state for next iteration
        state = state_prime
        reward = reward_prime

        if render:
            env.render()

        if done:
            break

    # record the "absorbing" state/experience
    experiences.append({
        'state': json.dumps(state.tolist()),
        'action': '',
        'reward': reward,
        'state_prime': '',
        'player': player,
        'player_prime': player_prime,
        'env_id': env.spec.id
    })

    return experiences

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('env_id', nargs='?', default='Tak3x3-points-v0')
    parser.add_argument('--mode', nargs='?', default='play', choices=['train', 'record', 'play'])
    parser.add_argument('--player', nargs='?', default='random', choices=['random', 'human', 'trained'])

    parser.add_argument('--iter', dest='iter', type=int, default=0)
    parser.add_argument('--limit', dest='limit', type=int, default=30000)

    parser.add_argument('--render', dest='render', action='store_true')
    parser.add_argument('--log', dest='log', action='store_true')
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.1)

    args = parser.parse_args()
    main(args)
