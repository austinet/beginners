import numpy as np
import time

import sys, os
sys.path.insert(0, '/Users/User/Documents/GitHub/Reinforcement')
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


def policy_evaluation(env, policy, gamma=1.0, theta=0.00001):
    # initialize all state values
    V = np.zeros(env.nS)
    print('value V: ', V)

    while True:
        delta = 0

        '''perform a "full backup";
            get all state values '''

        for s in range(env.nS):
            v = 0
            # look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # for each action, look at the possible next states
                for prob, next_state, reward, done in env.P[s][a]:
                    # calculate the expected value
                    # Vπ(s) = Σπ(a|s) * Σ{P(s'|s, a) * [r(s, a, s') + γV(s')]}
                    v += action_prob * prob * (reward + gamma * V[next_state])
            
            # Δ ← how much the value increased
            delta = max(delta, np.abs(v - V[s]))
            # replace an old value into the updated one
            V[s] = v

        # stop evaluation if Δ < θ
        if delta < theta:
            break

    return np.array(V)


def policy_iteration(env, gamma=1.0):
    # initialize all policies
    policy = np.ones([env.nS, env.nA]) / env.nA
    print('policy initiation: ', policy)

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

    while True:
        # evaluate the current policy
        V = policy_evaluation(env, policy, gamma)

        # found the optimal policy?
        policy_stable = True

        ''' policy improvement for each state '''

        for s in range(env.nS):
            print('---------- state {} ----------'.format(s))

            # the best action taken under the old policy
            chosen_a = np.argmax(policy[s])
            print('old best action: ', chosen_a)

            # greedily update the current policy
            action_values = one_step_lookahead(s, V)
            print('action values in state {}'.format(s), action_values)
            # π(s) ← argmax[Qπ(s, a)]
            best_a = np.argmax(action_values) # updated best action
            print('new best action: ', best_a)

            if chosen_a != best_a:
                policy_stable = False
            
            policy[s] = np.eye(env.nA)[best_a]
            print('the updated policy: ', policy[s])
            print()

        print('run time: {}s'.format(time.time() - start_time))

        # if the policy is stable, the algorithm found an optimal policy
        if policy_stable:
            return policy, V


opt_policy, opt_V = policy_iteration(env)
print()

print('<< the optimal policies π*(s) >>')
print(opt_policy)

print()

print('<< the optimal state values Vπ*(s) >>')
print(opt_V)