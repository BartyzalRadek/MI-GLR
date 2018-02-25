import gym
import numpy as np

env = gym.make('CartPole-v0')

theta = [ 0.62887362, -0.77817081, -0.28637888, -0.84578956]
state = env.reset()
ep_reward = 0
done = False
while not done:
    env.render()
    action = 0 if np.dot(theta, state) >= 0 else 1
    state, reward, done, _ = env.step(action)
    ep_reward += reward
    #print("Reward received = ", reward)
    #print("New state =", new_state)
print("Episode reward = ", ep_reward)