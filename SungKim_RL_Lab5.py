import gym
import numpy as np
import matplotlib.pyplot as plt
import random

method = input("Method of the action selection?  1.E-greedy  2.Add random noise")

env = gym.make('FrozenLake-v0')
env.render()

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
num_episodes = 2000
dis = .99 # Discount factor
learning_rate = .85

# Create lists to contain total rewards and steps per episode
rList = []

for i in range(num_episodes):

    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    e = 1./((i/100)+1)

    print(e)

    # The Q-Table learning algorithm
    while not done:
        # Choose an action by e-greedy
        if method==1:
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state,:])
        # Choose an action by greedily (with noise) picking from Q table
        elif method==2:
            action = np.argmax(Q[state,:]+np.random.randn(1,env.action_space.n)/(i+1))

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = (1-learning_rate)*Q[state,action]+learning_rate*(reward + dis*np.max(Q[new_state,:]))

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: "+str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)),rList, color="blue")
plt.show()