import numpy as np
from matplotlib import pyplot as plt
import time

from gymnastkit import *
from valuekit import *
from drawkit import *


class bellman_expectation:

    def __init__(self):
        self.env = Environment()
        self.agent = Agent([0, 0])
        self.max_step_num = 2


    def expect_state_values(self):
        env = self.env
        agent = self.agent
        max_step_num = self.max_step_num

        elapsed_time = []

        for max_step in range(max_step_num):
            value_table = np.zeros((env.rewards.shape[0], env.rewards.shape[1]))
            start_time = time.time()

            for i in range(env.rewards.shape[0]):
                for j in range(env.rewards.shape[1]):
                    agent.set_pos([i, j])
                    value_table[i, j] = state_value_func(env, agent, 0, max_step, 0)

            elapsed_time.append(time.time()-start_time)

            print('max_step_num = {} | total_time = {}(s)'.format(max_step, np.round(time.time()-start_time, 2)))

            show_v_table(np.round(value_table, 2), env)

        plt.plot(elapsed_time, 'o-k')
        plt.xlabel('max_down')
        plt.ylabel('time(s)')
        plt.legend()
        plt.show()


    def expect_action_values(self):
        env = self.env
        agent = self.agent
        max_step_num = self.max_step_num

        np.random.seed(0)

        for max_step in range(max_step_num):
            print('max_step = {}'.format(max_step))
            q_table = np.zeros((env.rewards.shape[0], env.rewards.shape[1], len(agent.action)))

            for i in range(env.rewards.shape[0]):
                for j in range(env.rewards.shape[1]):
                    for action in range(len(agent.action)):
                        agent.set_pos([i, j])
                        q_table[i, j, action] = action_value_func(env, agent, action, 0, max_step, 0)
        
            q = np.round(q_table, 2)
            print('Q-Table')
            show_q_table(q, env)
            print('high actions arrow')
            show_q_table_arrow(q, env)
            print()


belm_ex = bellman_expectation()
belm_ex.expect_action_values()
belm_ex.expect_state_values()