import argparse
import numpy as np
import pandas as pd
import copy
import json

import gym

import tak.env
from tak.board import Board
from tak.agent import RandomAgent, LearnerAgent, ValueFunction

def main(args):

    mode = args.mode
    env = gym.make(args.env_id)
    value_function = ValueFunction(size=env.board.size, height=env.board.height, id=env.spec.id)

    if mode == 'train':

        agent_white = get_player(args.player1, symbol=Board.WHITE, env=env, value_function=value_function)
        agent_black = get_player(args.player2, symbol=Board.BLACK, env=env, value_function=value_function)
        all_experiences = []

        for i in range(int(args.games)):

            if i % 20 == 0 and i > 0:
                print('Episode', i)

            loop_experiences = agent_environment_loop(env, agent_white, agent_black, render=False)
            all_experiences += loop_experiences

        experiences = pd.DataFrame(all_experiences)
        experiences.fillna(False, inplace=True)
        experiences.loc[:, 'state'] = experiences['state'].apply(convert)
        experiences.loc[:, 'state_prime'] = experiences['state_prime'].apply(convert)

        value_function.experience_replay(experiences, n_iter=args.iter, batch_size=args.batch_size)

    if mode == 'play':
        agent_white = get_player(args.player1, symbol=Board.WHITE, env=env, value_function=value_function)
        agent_black = get_player(args.player2, symbol=Board.BLACK, env=env, value_function=value_function)

        results = []
        for i in range(int(args.games)):

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

        print('\nAverage Score:', mean[0])
        print('Wins:', wins, 'Losses:', losses, 'Ties:', ties)

def get_player(player_type, symbol, env, value_function):
    if player_type == 'trained':
        return LearnerAgent(env=env, symbol=symbol, value_function=value_function)
    return RandomAgent(env=env, symbol=symbol, value_function=value_function)


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
            env.render(mode='human')

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
    parser.add_argument('--mode', nargs='?', default='play', choices=['train', 'play'])

    parser.add_argument('--player1', nargs='?', default='random', choices=['random', 'human', 'trained'])
    parser.add_argument('--player2', nargs='?', default='random', choices=['random', 'human', 'trained'])

    parser.add_argument('--iter', dest='iter', type=int, default=1)
    parser.add_argument('--games', dest='games', type=int, default=1)
    parser.add_argument('--render', dest='render', action='store_true')
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.1)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100)

    # train parameters
    parser.add_argument('--limit', dest='limit', type=int, default=1000000)

    args = parser.parse_args()
    main(args)
