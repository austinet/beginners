# Bellman Equations Kit

import numpy as np

def state_value_func(env, agent, G, max_step, current_step):
    '''
    parameters
    ---
        env: environment
        agent: agent
        G: return; sum of total rewards
        max_step: the number of steps to calculate the value
        current_step: the current calcuating step
    '''

    # discount factor
    gamma = 0.9

    # if the current position is 'goal'
    if env.reward_list_char[agent.pos[0]][agent.pos[1]] == 'goal':
        return env.goal # EXIT FUNCTION

    ''' if the current position is not goal,
    calculate the value of the state '''

    # 1. value of the last state
    if (max_step == current_step):
        current_pos = agent.get_pos()

        # sum of rewards of all possible actions
        for i in range(len(agent.action)):
            agent.set_pos(current_pos)
            _, reward, _ = env.move(agent, i)
            G += agent.action_pr[i] * reward

        return G # EXIT FUNCTION

    # 2. value of the entire states except the last one
    else:
        current_pos = agent.get_pos()

        # i: the number of possible actions
        for i in range(len(agent.action)):
            observation, reward, done = env.move(agent, i)

            G += agent.action_pr[i] * reward

            # if the next step is out of the field:
            if done == True:
                if observation[0] < 0 or observation[0] >= env.rewards.shape[0] or \
                observation[1] < 0 or observation[1] >= env.rewards.shape[1]:
                    agent.set_pos(current_pos) # get back to the previous position

            next_value = state_value_func(env, agent, 0, max_step, current_step+1) # RECURSIVE
            G += agent.action_pr[i] * gamma * next_value

            agent.set_pos(current_pos) # get back to the previous position

            '''
            a reason why bring the position back:
                change of the position is allowed only in env.move();
                in state_value_func(), nothing but 'observation' is allowed.
            '''

        return G


def action_value_func(env, agent, act, G, max_step, current_step):
    '''
    parameters
    ---
        env: environment
        agent: agent
        G: return; sum of total rewards
        max_step: the number of steps to calculate the value
        current_step: the current calcuating step
    '''

    # discount factor
    gamma = 0.9

    # check if the current position is 'goal'
    if env.reward_list_char[agent.pos[0]][agent.pos[1]] == 'goal':
        return env.goal # EXIT FUNCTION

    ''' if the current position is not goal,
    calculate the value of the state '''

    # 1. value of the last state
    if(max_step == current_step):
        _, reward, _ = env.move(agent, act)
        G += agent.action_pr[act] * reward
        return G # EXIT FUCNTION

    # 2. value of the entire states except the last one
    else:
        current_pos = agent.get_pos()
        observation, reward, done = env.move(agent, act)
        G += agent.action_pr[act] * reward

        # if the next step is out of the field:
        if done == True:
            if observation[0] < 0 or observation[0] >= env.rewards.shape[0] or \
            observation[1] < 0 or observation[1] >= env.rewards.shape[1]:
                agent.set_pos(current_pos) # get back to the previous position

        current_pos = agent.get_pos() # FETCH the current position

        for i in range(len(agent.action)):
            agent.set_pos(current_pos) # save the current position
            next_value = action_value_func(env, agent, i, 0, max_step, current_step+1) # RECURSIVE
            G += agent.action_pr[i] * gamma * next_value

        return G

