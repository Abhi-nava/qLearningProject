# without epsilon takes 300 episodes
import gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=- 2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0 and episode != 0:
        print(episode)
        env = gym.make("MountainCar-v0", render_mode="human")
    elif episode % SHOW_EVERY != 0:
        env = gym.make("MountainCar-v0")

    discrete_state = get_discrete_state(env.reset()[0])
    done = False
    truncated = False
    while not done and not truncated:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, truncated, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

env.close()