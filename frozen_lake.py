import gymnasium as gym
from gymnasium.wrappers import TransformReward
import qlearning
import numpy as np
# https://gymnasium.farama.org/environments/toy_text/frozen_lake/
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode=None)
env = TransformReward(env, lambda r: 1 if r == 1 else r-0.04)

observation, info = env.reset(seed=42)
action = env.action_space.sample()
print("Starting")
terminated_i = 0
Q_table = qlearning.QTable(0.99, 0.1, 0.5, env.action_space.n, env.observation_space.n)
### TRAINING LOOP
# In this loop your agent will be trained without showing the run
# To stop this training loop you can pres ctrl+c in the terminal. This will start the testing loop below
# You can implement a stop criteria if you want
try:
    convergence = 0.00001
    stable_episodes = 0
    previous_Q_table = Q_table.Q.copy()
    while True:
        observation, reward, terminated, truncated, info = env.step(action)
        action = Q_table.update_Q(observation, reward)  ### <-- This line samples a random action for from the environment. Replace this with your Qlearning algorithm ###

        if terminated or truncated:         ### <-- This condition triggers if the run is terminated, as it the agent reaches a goal or falls into a hole
            terminated_i = terminated_i + 1


            diffrence = np.abs(Q_table.Q - previous_Q_table).max()
            if diffrence < convergence:
                stable_episodes += 1
            else:
                stable_episodes = 0
            if stable_episodes >= 1000:
                print("Q-table has converged after", terminated_i, "episodes.")
                break

            previous_Q_table = Q_table.Q.copy()


            if reward > 0:                  ### <-- This condition triggers when the agent finds the goal. You are free to debug and print how you want
                print("Found goal at iteration", terminated_i)

            observation, info = env.reset()
except KeyboardInterrupt:
    print("Run ended")
    env.close()


### TESTING LOOP
# Put your algorithm for taking the best action here
# To stop this testing loop, press ctrl+c in the terminal.
env_test = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode='human')
observation, info = env_test.reset(seed=42)


Q_table.set_epsilon_value(0.0)  # Set epsilon to 0 to always take the best action
episodes = 0
try:
    while True:
        observation, reward, terminated, truncated, info = env_test.step(action)
        action = Q_table.update_Q(observation, reward)  ### <-- This line samples a random action for from the environment. Replace this with your optimal action calculation ###

        if terminated or truncated:
            episodes += 1
            print("Episode", episodes, "ended with reward", reward)
            if episodes >= 10:
                print("Test ended after 15 episodes")
                break
            observation, info = env_test.reset()

except KeyboardInterrupt:
    print("Test ended")
    env_test.close()