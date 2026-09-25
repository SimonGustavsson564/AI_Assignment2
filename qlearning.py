import numpy as np

np.seterr(all='raise')

### This file contains suggested skeleton code if you do not know how to start ###
##  You do not have to follow this skeleton code, you can use your own structure ##


class QTable:
    def __init__(self, _gamma, _alpha, _epsilon, _action_size, _state_size):

        self.alpha = _alpha
        self.gamma = _gamma
        self.epsilon = _epsilon
        self.action_size = _action_size
        self.initial_array_size = _state_size


        # s, a, r , the previous state, action, and reward, initially null
        self.previous_Q_index = np.nan
        self.previous_action = np.nan
        self.previous_reward = np.nan

        # Initiate Q-tables here
        self.Q = np.zeros((_state_size, _action_size))  # Rows = states, columns = actions

    def get_Q_index(self, coded_state):
        return coded_state  # Return the index of a row in the Qtable ?????????

    def get_Q_actions(self, Q_index):
        return self.Q[Q_index]  # Return action for a row in the Qtable

    def set_Q(self, Q_index, action, new_Q):
        # Set calculated Q value for a state action pair
        self.Q[Q_index, action] = new_Q

    def get_max_Q(self, Q_index):
        return np.max(self.Q[Q_index])  # Maximum Q value in a row

    def epsilon_greedy(self, Q_index):

        # Random action with probability epsilon, otherwise the best one
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return self.get_best_action(Q_index)

    def get_best_action(self, current_state):

        return np.argmax(self.Q[current_state]) # Best action

    def start_episode(self, start_state):
        # New episode: reset s, a, r
        next_action = self.epsilon_greedy(start_state)

        self.previous_Q_index = self.get_Q_index(start_state)
        self.previous_action = next_action
        self.previous_reward = np.nan

        return next_action

    def update_Q(self, current_state, current_reward, terminated=False):
        # Algorithm used from "Artificial Intelligence A Modern Approach" by Stuart Russell and Peter Norvig
        # Q-Learning-Agent, page 844, figure 21.8

        if not np.isnan(self.previous_Q_index):

            old_Q_value = self.get_Q_actions(self.previous_Q_index)[self.previous_action]
            if terminated:
                # Terminal state: no future reward
                target = current_reward
            else:
                # Reward + discounted best Q of the next state
                target = current_reward + self.gamma * self.get_max_Q(current_state)
            # Move Q a step (alpha) towards the target
            new_Q_value = old_Q_value + self.alpha * (target - old_Q_value)
            self.set_Q(self.previous_Q_index, self.previous_action, new_Q_value)

        if terminated:
            return None  # Episode over, no next action

        next_action = self.epsilon_greedy(current_state)

        # Remember s, a, r for the next update
        self.previous_Q_index = self.get_Q_index(current_state)
        self.previous_action = next_action
        self.previous_reward = current_reward

        return next_action  # Action to take for next step AFTER epsilon greedy has been used
    def set_epsilon_value(self, epsilon_value):
        self.epsilon = epsilon_value
        return 
