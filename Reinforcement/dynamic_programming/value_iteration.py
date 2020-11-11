import numpy as np
import time

import sys, os
sys.path.insert(0, '/Users/austin/OneDrive/Repositories/Reinforcement')
from dynamic_programming.gridworld import GridworldEnv

'''
Definition

    1. S, A, V
        S/s for state;
        A/a for action;
        V/v for value.

    2. policy
        (s, a) shaped matrix representing the policy of each state.
        if policy shaped (9, 4), there are 9 states and 4 actions in each state.

    3. env
        OpenAI environment.
        
        env.nA: a number of actions in each state.
        env.nS: number of entire states in environment.
        env.P[s][a]: a list of tuples (prob, next_state, reward, done).
        'env.P' represents the transition probabilities of the environment.

    4. hyperparameters
        theta: a threshold to stop the algroithm.
        gamma: discount factor.
'''

env = GridworldEnv()
print('number of actions(nA): ', env.nA)
print('number of states(nS): ', env.nS)


def value_iteration(env, gamma=1.0, theta=0.00001):

    def one_step_lookahead(state, V):
        # initialize all actions
        A = np.zeros(env.nA)

        # get action values
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                # Qπ(s, a) = Σ{P(s'|s, a) * [r(s, a, s') + γV(s')]}
                A[a] += prob * (reward + gamma * V[next_state])

        return A

    start_time = time.time()

    # initialize all states
    V = np.zeros(env.nS)

    k = 0 # number of iteration
    while True:
        delta = 0

        ''' value iteration '''

        print()
        print('===========')
        print('iteration {}'.format(k))
        print('===========')
        print()

        # get the best action value
        # state value ← the best action value
        for s in range(env.nS):
            print('---------- state {} ----------'.format(s))

            A = one_step_lookahead(s, V)
            # V(s) = max{P(s'|s, a)[r(s, a, s') + γV(s')]}
            best_action_value = np.max(A)

            print('all action values: ', A)
            print('the best action value: ', best_action_value)

            # Δ ← how much the value increased
            delta = max(delta, np.abs(best_action_value - V[s]))
            # state value ← the best action value
            V[s] = best_action_value

        k += 1

        if delta < theta:
            break

    print()
    print('<< the optimal state values Vπ*(s) >>')
    print(V)
    print()

    ''' the optimal policy extraction '''

    # initialize all policies
    policy = np.zeros([env.nS, env.nA])
    
    for s in range(env.nS):
        print('---------- state {} ----------'.format(s))

        # update the current policy
        # V = best_action_value
        A = one_step_lookahead(s, V)
        # π*(s) ← argmax[Qπ(s, a)]
        best_action = np.argmax(A)
        print('the best action: ', best_action)

        # deterministic world: let the agent always takes the best action
        policy[s, best_action] = 1.0
        print('policy: ', policy)

    print('run time: {}s'.format(time.time() - start_time))

    return policy, V


policy, v = value_iteration(env)
print()

print('<< the optimal policies π*(s) >>')
print(policy)
