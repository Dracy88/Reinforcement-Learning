import numpy as np
import random as rnd
rnd.seed(42)
np.random.seed(42)


class Agent:
    def __init__(self, state_size, action_size, n_episodes):
        """
        :param state_size: specific the size of a single state
        :param action_size: specific the number of all the possible actions
        :param n_episodes: specific the number of episodes for our training
        """
        self.state_size = state_size  # Specific the size of observation coming from the environment
        self.action_size = action_size  # Specific the number of the available actions (up, down, left, right)

        self.gamma = 0.98  # Indicate how much important are the future rewards
        self.epsilon = 1.0  # Indicates the probability to do a random choice instead a smart choice
        self.epsilon_min = 0.01  # Indicates the minimum probability to do a random choice
        self.epsilon_decay = self.epsilon / n_episodes  #
        self.firstIter = True  # Monitor if we are in our first action
        self.discount = 0.8  # Discount factor for the Q-value formula
        self.Q_table = np.zeros((state_size[0], state_size[1], action_size))

    def act(self, state, episode, train_mode):
        # Decay epsilon
        self.epsilon = (np.exp(-0.01 * episode))

        # Determine if the next action will be a random move or with strategy
        if train_mode and self.epsilon > np.random.rand():  # Do a random action (Exploration)
            return rnd.randrange(self.action_size)  # Return a random action only in train phase
        else:
            return np.argmax(self.Q_table[state])  # Return the best action for the current state (Exploitation)

    def learn(self, state, action, next_state, reward):
        # Getting the actual x,y coordinates of the current state
        y = state[0]
        x = state[1]
        # Getting the new x,y coordinates of the next state that we had
        n_y = next_state[0]
        n_x = next_state[1]

        # Q[s, a] = Q[s, a] + ALPHA * (reward + GAMMA * mx.nd.max(Q[observation, :]) - Q[s, a])
        self.Q_table[y, x, action] = \
            self.Q_table[y, x, action] + \
            self.discount * (reward + self.gamma * np.amax(self.Q_table[n_y, n_x, :]) - self.Q_table[y, x, action])

    def get_q_table(self):
        return self.Q_table

















