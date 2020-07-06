from environment import Environment
from agent import Agent
import matplotlib.pyplot as plt

n_episodes = 500
env = Environment()
agent = Agent(state_size=env.get_state_size(), action_size=env.get_actions_n(), n_episodes=n_episodes)
actions = env.get_available_actions()  # Getting the available action of the environment
episode_rewards = []

for ep in range(n_episodes):
    done = False
    ep_reward = reward = step = 0
    state = env.reset()

    while not done:
        action = agent.act(state, ep, train_mode=True)  # The agent choose an action based on the current state
        next_state, reward, done = env.step(actions[action])  # Getting the next state and reward based on the action choose
        agent.learn(state, action, next_state, reward)

        ep_reward += reward
        step += 1
        state = next_state

    episode_rewards.append(ep_reward)


# Showing the learning graph
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(episode_rewards, '-g', label='reward')
ax1.set_xlabel("episode")
ax1.set_ylabel("reward")
ax1.legend(loc=2)
plt.title("Training Progress")
plt.show()

# Testing our agent
for ep in range(1):
    done = False
    reward = 0
    state = env.reset()

    while not done:
        action = agent.act(state, ep, train_mode=False)  # The agent choose an action based on the current state
        next_state, reward, done = env.step(actions[action])  # Getting the next state and reward based on the action choose
        state = next_state

    env.render()  # Showing a step by step presentation of a episode

