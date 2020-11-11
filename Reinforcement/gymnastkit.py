import numpy as np


class Environment:

    # set rewards
    cliff = -3
    road = -1
    goal = 1

    # set the destination
    destination = [2, 2]

    # discrete field (num)
    reward_list_num = [[road, road, road],
                       [road, road, road],
                       [road, road, goal]]

    # discrete field (char)
    reward_list_char = [['road', 'road', 'road'],
                        ['road', 'road', 'road'],
                        ['road', 'road', 'goal']]


    def __init__(self):
        self.rewards = np.asarray(self.reward_list_num)


    def move(self, agent, act):
        # end of episode?
        done = False

        # new_pos: renewed position after action
        # agent.pos = Agent().self.pos
        # agent.action[action] = Agent().action[i]
        new_pos = agent.pos + agent.action[act]

        # if the current position is 'goal'
        if self.reward_list_char[agent.pos[0]][agent.pos[1]] == 'goal':
            reward = self.goal
            observation = agent.set_pos(agent.pos)
            done = True # END OF EPISODE

        # if the new position is 'cliff'
        elif new_pos[0] < 0 or new_pos[0] >= self.rewards.shape[0] \
        or new_pos[1] < 0 or new_pos[1] >= self.rewards.shape[1]:
            reward = self.cliff
            observation = agent.set_pos(agent.pos)
            done = True # END OF EPISODE

        # if the new position is 'road'
        else:
            observation = agent.set_pos(new_pos)
            reward = self.rewards[observation[0], observation[1]] # EPISODE ONGOING

        return observation, reward, done


class Agent:
    # available actions for the agent
    # up, right, down, left
    action = np.array([[-1, 0], [0,1], [1, 0], [0, -1]])

    # probabilities for each action
    action_pr = np.array([0.25, 0.25, 0.25, 0.25])


    def __init__(self):
        self.pos = (0, 0)

    
    # record the current agent's position
    def set_pos(self, position):
        self.pos = position
        return self.pos


    # fetch the current agent's position
    def get_pos(self):
        return self.pos
