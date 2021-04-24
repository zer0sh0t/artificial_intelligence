import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
EPISODES = 25000
# MAX_STEPS = 2000
SHOW_EVERY = 2000
EPSILON = 0.65
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
EPSILON_DECAY_VALUE = EPSILON / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
LEARNING_RATE = 0.45
GAMMA = 0.8
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low)/DISCRETE_OS_SIZE
# print("os win size : " + str(discrete_os_win_size))

Q = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# print(Q.shape)

# testing
# state1 = np.array([0.078, -0.00045])
# state2 = env.reset()
# state3 = env.observation_space.sample()


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

    # testing
# print("state1 : " + str(state1))
# print("state2 : " + str(state2))
# print("state3 : " + str(state3))
#
# ds1 = get_discrete_state(state1)
# ds2 = get_discrete_state(state2)
# ds3 = get_discrete_state(state3)
#
# print("ds1 : " + str(ds1))
# print("ds2 : " + str(ds2))
# print("ds3 : " + str(ds3))
#
# print("Q1 : " + str(Q[ds1]))
# print("Q2 : " + str(Q[ds2]))
# print("Q3 : " + str(Q[ds3]))
# print(Q[(5,-2)])


for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print(episode)
        RENDER = True
    else:
        RENDER = False

    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:

        if np.random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[discrete_state])
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if RENDER:
            env.render()

        if not done:
            Q[discrete_state + (action,)] = Q[discrete_state + (action,)] + LEARNING_RATE * (
                reward + GAMMA * np.max(Q[new_discrete_state]) - Q[discrete_state + (action,)])

        if new_state[0] >= env.goal_position:
            Q[discrete_state + (action,)] = 0
            print("We finished the level in " + str(episode) + " episode")

        discrete_state = new_discrete_state
        ep_rewards.append(episode_reward)

    EPSILON -= EPSILON_DECAY_VALUE
    if episode % SHOW_EVERY == 0:
        average_reward = (
            sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(
            f"Episode:{ episode } | Average:{ average_reward } | Minimum:{ min(ep_rewards[-SHOW_EVERY:]) } | Maximum:{ max(ep_rewards[-SHOW_EVERY:]) }")

    env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
