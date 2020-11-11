# Monte Carlo

import numpy as np

def generate_episode(env, agent, first_visit):
    gamma = 0.09

    episode = []

    visit = np.zeros((env.rewards.shape[0], env.rewards.shape[1]))

    i = np.random.randint(0, env.rewards.shape[0])
    j = np.random.randint(0, env.rewards.shape[1])
    agent.set_pos([i, j])

    G = 0

    step = 0
    max_step = 100

    for k in range(max_step):
        pos = agent.get_pos()
        action = np.random.randint(0, len(agent.action))
        observation, reward, done = env.move(agent, action)

        if first_visit:
            if visit[pos[0], pos[1]] == 0:
                G += gamma ** (step) * reward
                visit[pos[0], pos[1]] = 1
                step += 1
                episode.append((pos, action, reward))
        else:
            G += gamma ** (step) * reward
            step += 1
            episode.append((pos, action, reward))

        if done == True:
            break
    
    return i, j, G, episode

# ======================================================

from tqdm import tqdm
from gymnastkit import *
from drawkit import *

np.random.seed(0)

env = Environment()
agent = Agent()

v_table = np.zeros((env.rewards.shape[0], env.rewards.shape[1]))
v_start = np.zeros((env.rewards.shape[0], env.rewards.shape[1]))
v_success = np.zeros((env.rewards.shape[0], env.rewards.shape[1]))

return_s = [[[] for j in range(env.rewards.shape[1])] for i in range(env.rewards.shape[0])]

max_episode = 100000

first_visit = False

if first_visit:
    print('start first visit MC')
else:
    print('start every visit MC')
print()

for epi in tqdm(range(max_episode)):
    i, j, G, episode = generate_episode(env, agent, first_visit)

    return_s[i][j].append(G)

    episode_count = len(return_s[i][j])

    total_G = np.sum(return_s[i][j])
    
    v_table[i, j] = total_G / episode_count

    if episode[-1][2] == 1:
        v_success[i, j] += 1

for i in range(env.rewards.shape[0]):
    for j in range(env.rewards.shape[1]):
        v_start[i, j] = len(return_s[i][j])

print('V(s)')
show_v_table(np.round(v_table, 2), env)

print('v_start_count(s)')
show_v_table(np.round(v_start, 2), env)

print('v_success_pr(s)')
show_v_table(np.round(v_success / v_start, 2), env)