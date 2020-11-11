
''' dynaprokit :: dynamic programming method '''

import numpy as np
import time
import copy

from gymnastkit import *
from drawkit import *

class PolicyIteration:

    ''' policy iteration '''

    def __init__(self, env, agent, gamma=0.9):
        self.env = env
        self.agent = agent
        self.gamma = gamma


    def prediction(self, v_table, policy):

        ''' policy evaluation '''

        env = self.env
        agent = self.agent
        gamma = self.gamma

        while(True):
            # Δ ← 0
            delta = 0

            # v ← V(s)
            tmp_v = copy.deepcopy(v_table)

            # for each s ∊ set S,
            for i in range(env.rewards.shape[0]): # shape[0] = number of rows(horizontal)
                for j in range(env.rewards.shape[1]): # shape[1] = number of columns(vertical)
                    # current location of the agent
                    agent.set_pos([i, j])

                    # action a = π(s)
                    action = policy[i, j]
                    observation, reward, _ = env.move(agent, action) # move the agent in order to policy π

                    # v_table[i, j] = V(s)
                    # reward = r(s, a, s'); value of V(s')
                    # gamma * v_table[observation[0], observation[1]] = γV(s')
                    v_table[i, j] = reward + gamma * v_table[observation[0], observation[1]]

            # Δ ← max(Δ, |v - V(s)|)
            delta = np.max([delta, np.max(np.abs(tmp_v - v_table))])

            # iterate until Δ ≤ |v - V(s)|
            if delta < 0.000001:
                break

        return v_table, delta


    def control(self, v_table, policy):

        ''' policy improvement '''

        env = self.env
        agent = self.agent
        gamma = self.gamma
        policy_stable = True

        # for each s ∊ set S,
        for i in range(env.rewards.shape[0]):
            for j in range(env.rewards.shape[1]):
                # old action ← π(s)
                old_action = policy[i, j]

                # argmax
                tmp_action = 0
                tmp_value = -1e+10

                for action in range(len(agent.action)):
                    agent.set_pos([i, j])
                    observation, reward, _ = env.move(agent, action)

                    if tmp_value < reward + gamma * v_table[observation[0], observation[1]]:
                        tmp_action = action
                        tmp_value = reward + gamma * v_table[observation[0], observation[1]]

                # if old action != current action
                # policy is NOT STABLE yet
                # so keep repeating
                if old_action != tmp_action:
                    policy_stable = False
                
                policy[i, j] = tmp_action # renew the policy

        return policy, policy_stable


np.random.seed(0)
env = Environment()
agent = Agent([0, 0])

pol_iter = PolicyIteration(env, agent)

v_table = np.random.rand(env.rewards.shape[0], env.rewards.shape[1])
policy = np.random.randint(0, 4, (env.rewards.shape[0], env.rewards.shape[1]))

print('v table')
print(v_table)
print()
print('policy')
print(policy)

print('Initial random V(S)')
show_v_table(np.round(v_table, 2), env)
print()
print('Initial random Policy π0(S)')
show_policy(policy, env)
print('start policy iteration')

start_time = time.time()
max_iter_num = 20000

for iter in range(max_iter_num):
    # policy evaluation
    v_table, delta = pol_iter.prediction(v_table, policy)

    # print out
    print('Vπ{0:}(S) delta = {1:.10f}'.format(iter, delta))
    show_v_table(np.round(v_table, 2), env)
    print()

    # policy improvement
    policy, policy_stable = pol_iter.control(v_table, policy)

    # print out
    print('policy π{}(S)'.format(iter+1))
    show_policy(policy, env)

    # if old action != action:
    # keep iterating
    if(policy_stable == True):
        break

print('total_time = {}'.format(time.time() - start_time))