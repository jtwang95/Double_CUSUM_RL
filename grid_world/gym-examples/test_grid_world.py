import gym as gym
import numpy as np
import time, gym_examples

# env = gym.make("gym_examples/GridWorld-v0", render_mode="human")
# observation, info = env.reset(seed=42)

# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()
#     env.render()
#     time.sleep(0.01)
# env.close()

SIZE_GRID = 4
NUM = 100
RENDER_MODE = "human"
CHANGE_POINT = 25
CENSOR_TIME = 50
env = gym.make("gym_examples/GridWorld-v0",
               render_mode=RENDER_MODE,
               size=SIZE_GRID,
               change_point=CHANGE_POINT,
               censor_time=CENSOR_TIME)
observation, info = env.reset(seed=42)

# steps_to_terminal = np.zeros([NUM])
states = np.zeros([NUM, CENSOR_TIME + 1, 1], dtype=int)
actions = np.zeros([NUM, CENSOR_TIME], dtype=int)
rewards = np.zeros([NUM, CENSOR_TIME], dtype=int)

for i in range(NUM):
    states_tmp = np.zeros([CENSOR_TIME + 1], dtype=int) * -1
    actions_tmp = np.zeros([CENSOR_TIME], dtype=int) * -1
    rewards_tmp = np.zeros([CENSOR_TIME], dtype=int) * -1
    observation, info = env.reset(seed=i)
    current_t = 0
    states_tmp[current_t] = info["state"]
    for _ in range(10000):
        current_t = current_t + 1
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        states_tmp[current_t] = info["state"]
        rewards_tmp[current_t - 1] = reward
        actions_tmp[current_t - 1] = action
        if terminated:
            observation, info = env.reset(seed=i)
            current_t = 0
            states_tmp[current_t] = info["state"]

        if truncated:
            states[i] = states_tmp.reshape([-1, 1])
            rewards[i] = rewards_tmp
            actions[i] = actions_tmp
            # steps_to_terminal[i] = info["steps"]
            break
env.close()
print(states)
print(actions)
print(np.sum(rewards == 1))
# print(np.mean(steps_to_terminal))
# print(np.mean(steps_to_terminal >= 50))

# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.histplot(steps_to_terminal)
# plt.show()
