import gymnasium as gym
import qlearning
import numpy as np

# https://gymnasium.farama.org/environments/classic_control/acrobot/
env = gym.make('Acrobot-v1', render_mode=None)

# Or
# https://gymnasium.farama.org/environments/box2d/lunar_lander/
#env = gym.make("LunarLander-v2", render_mode=None)


### FUNCTION APPROXIMATOR
# Continuous observation -> Q-table row index
BINS = [8, 8, 8, 8]  # theta1, theta2, velocity1, velocity2
LIMITS = [(-np.pi, np.pi), (-np.pi, np.pi), (-6, 6), (-12, 12)]  # Min/max for each value
STATE_SIZE = int(np.prod(BINS))

def discretize(observation):
    # Angles from cos/sin
    theta1 = np.arctan2(observation[1], observation[0])
    theta2 = np.arctan2(observation[3], observation[2])
    values = [theta1, theta2, observation[4], observation[5]]

    index = 0
    for value, (low, high), bins in zip(values, LIMITS, BINS):
        bin_number = int((value - low) / (high - low) * bins)  # Which interval the value is in
        bin_number = min(max(bin_number, 0), bins - 1)          # Keep inside 0..bins-1
        index = index * bins + bin_number                       # Combine all intervals into one number
    return index


EPISODES = 2000  # Stop criteria
Q_table = qlearning.QTable(0.99, 0.1, 1.0, env.action_space.n, STATE_SIZE)  # gamma, alpha, epsilon, actions, states

observation, info = env.reset()
action = Q_table.start_episode(discretize(observation))
print("Starting")
terminated_i = 0
steps = 0
episode_steps = []

### TRAINING LOOP
# In this loop your agent will be trained without showing the run
# To stop this training loop you can pres ctrl+c in the terminal. This will start the testing loop below
# You can implement a stop criteria if you want
try:
    while terminated_i < EPISODES:
        observation, reward, terminated, truncated, info = env.step(action)
        steps = steps + 1
        action = Q_table.update_Q(discretize(observation), reward, terminated)

        if terminated or truncated:
            terminated_i = terminated_i + 1
            episode_steps.append(steps)
            steps = 0

            # Epsilon decay: 1.0 -> 0.01
            Q_table.set_epsilon_value(max(0.01, 1.0 - terminated_i / (0.8 * EPISODES)))

            if terminated_i % 100 == 0:
                print("Episode", terminated_i, "- average steps last 100 episodes:", np.mean(episode_steps[-100:]), "- epsilon:", round(Q_table.epsilon, 2))

            observation, info = env.reset()
            action = Q_table.start_episode(discretize(observation))  # First action of the new episode
except KeyboardInterrupt:
    print("Run ended")
env.close()


### TESTING LOOP
# Put your algorithm for taking the best action here
# To stop this testing loop, press ctrl+c in the terminal.
env_test = gym.make('Acrobot-v1', render_mode='human')

#env_test = gym.make("LunarLander-v2", render_mode='human')

observation, info = env_test.reset()
action = Q_table.get_best_action(discretize(observation))
episodes = 0
steps = 0
try:
    while True:
        observation, reward, terminated, truncated, info = env_test.step(action)
        steps = steps + 1
        action = Q_table.get_best_action(discretize(observation))  # Best action only, no learning

        if terminated or truncated:
            episodes = episodes + 1
            print("Episode", episodes, "reached the goal" if terminated else "did NOT reach the goal", "in", steps, "steps")
            if episodes >= 10:
                break
            steps = 0
            observation, info = env_test.reset()
            action = Q_table.get_best_action(discretize(observation))

except KeyboardInterrupt:
    print("Test ended")
env_test.close()
