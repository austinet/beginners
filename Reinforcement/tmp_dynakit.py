# Dynamic Programming Kit

import numpy as np
import copy

from gymnastkit import *

class DynaProgramming:

    ''' dynamic programming method '''

    def __init__(self):
        self.env = Environment()
        self.agent = Agent([0, 0])
        self.gamma = 0.9
        self.np.random.seed(0)

    
    def policy_evaluation(self, env, agent, v_table, policy):
        # shape[0] = number of rows(horizontal)
        # shape[1] = number of columns(vertical)
        v_table = np.zeros((env.rewards.shape[0], env.rewards.shape[1]))
        step = 1

        while(True):
            delta = 0

            temp_v = copy.deepcopy(v_table)

            for i in range(env.rewards.shape[0]):
                for j in range(env.rewards.shape[1]):
                    G = 0

                    for action in range(len(agent.action)):
                        agent.set_pos([i, j])
                        observation, reward, done = env.move(agent, action)

                        # state value function
                        G += agent.action_pr[action] * (reward + gamma * v_table[observation[0], observation[1]])

                    v_table[i, j] = G

            delta = np.max([delta, np.max(np.abs(temp_v - v_table))])

            step += 1

            if delta < 0.000001:
                break