import gym as gym
import numpy as np
import gym_examples


def generate_grid_world_trajectories(num, size_grid, change_point, censor_time,
                                     seed,policy=None):
    env = gym.make("gym_examples/GridWorld-v0",
                   render_mode=None,
                   size=size_grid,
                   change_point=change_point,
                   censor_time=censor_time)
    observation, info = env.reset(seed=seed)
    states = np.zeros([num, censor_time + 1, 1], dtype=int)
    actions = np.zeros([num, censor_time], dtype=int)
    rewards = np.zeros([num, censor_time], dtype=int)

    for i in range(num):
        states_tmp = np.ones([censor_time + 1], dtype=int) * -1
        actions_tmp = np.ones([censor_time], dtype=int) * -1
        rewards_tmp = np.ones([censor_time], dtype=int) * -1
        observation, info = env.reset(seed=seed * 10000 + i)
        current_t = 0
        states_tmp[current_t] = info["state"]
        for _ in range(10000):
            current_t = current_t + 1
            if not policy:
                action = env.action_space.sample()
            else:
                action = policy.act(states_tmp[current_t-1],current_t)
            observation, reward, terminated, truncated, info = env.step(action)
            states_tmp[current_t] = info["state"]
            rewards_tmp[current_t - 1] = reward
            actions_tmp[current_t - 1] = action

            if truncated:
                states[i] = states_tmp.reshape([-1, 1])
                rewards[i] = rewards_tmp
                actions[i] = actions_tmp
                break
    env.close()
    return states, actions, rewards


if __name__ == "__main__":
    class QTable:
        def __init__(self,num_states,num_actions) -> None:
            self.num_states = num_states
            self.num_actions = num_actions
            self.q_table = np.random.normal(size=[self.num_states,self.num_actions])

        def act(self,s):
            print(s)
            return np.argmax(self.q_table[s,:])

    states, actions, rewards = generate_grid_world_trajectories(
        num=10, size_grid=4, change_point=25, censor_time=50,policy=None)
    print(rewards)
