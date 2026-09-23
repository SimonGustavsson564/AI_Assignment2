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
        self.previous_Q_index = np.NaN
        self.previous_action = np.NaN
        self.previous_reward = np.NaN

        # Initiate Q-tables here


    def get_Q_index(self, coded_state):
        return # Return the index of a row in the Qtable

    def get_Q_actions(self, Q_index):
        return # Return action for a row in the Qtable

    def set_Q(self, Q_index, action, new_Q):
        # Set calculated Q value for a state action pair

    def get_max_Q(self, Q_index):
        return # Maxumum Q value in a row

    def epsilon_greedy(self, Q_index):

        return # Either best action or random action
        

    def get_best_action(self, current_state):

        return # Best action

    def update_Q(self, current_state, current_reward):
        # Algorithm used from "Artificial Intelligence A Modern Approach" by Stuart Russell and Peter Norvig
        # Q-Learning-Agent, page 844, figure 21.8

        return # Action to take for next step AFTER epsilon greedy has been used
