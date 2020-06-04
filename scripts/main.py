import time
import gym
import random
from IPython.display import clear_output
import numpy as np

env = gym.make("FrozenLake-v0")

action_column_size = env.action_space.n
observation_column_size = env.observation_space.n

q_table = np.zeros((observation_column_size, action_column_size))

num_ep = 10000
num_step = 100
discount_rate = 0.99
learning_rate = 0.1
exploration_rate = 1
max_exp_rate = 1
min_exp_rate = 0.01
exp_rate_decay = 0.001

all_ep_reward = []

for episode in range(num_ep):
    state = env.reset()
    done = False
    current_ep_reward = 0

    for step in range(num_step):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                 learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        current_ep_reward += reward

        if done == True:
            break

    exploration_rate = min_exp_rate + \
                       (max_exp_rate - min_exp_rate) * np.exp(-exp_rate_decay*episode)

    all_ep_reward.append(current_ep_reward)

rewards_per_thousand_episodes = np.split(np.array(all_ep_reward),num_ep/1000)
count = 1000

print("-------------Average reward per thousand episodes-------------\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

print(q_table)