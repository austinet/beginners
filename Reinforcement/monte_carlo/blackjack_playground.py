import numpy as np
import time

import sys, os
sys.path.insert(0, '/Users/austin/OneDrive/Repositories/Reinforcement')
from monte_carlo.casinoworld import BlackjackEnv

env = BlackjackEnv()


def print_observation(observation):
    player_score, dealer_score, usable_ace = observation
    print('player score: {} (usable Ace: {}), dealer score: {}'.format(player_score, usable_ace, dealer_score))


def strategy(observation):
    player_score, dealer_score, usable_ace = observation

    # Stick(action 0) if the score > 20, hit(action 1) otherwise
    return 0 if player_score >= 20 else 1


for i_episode in range(20):
    observation = env.reset() # BlackjackEnv.reset()
    
    for t in range(100):
        print_observation(observation)
        action = strategy(observation)

        # print 'Stick' if action == 0, print 'Hit' if action == 1
        print('taking action: {}'.format(['Stick', 'Hit'][action]))

        observation, reward, done, _ = env.step(action) # BlackjackEnv.step()

        if done:
            print_observation(observation)
            print('game done.. reward: {}\n'.format(float(reward)))
            break