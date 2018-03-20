import gym
import numpy as np

'''
This example shows a solution for CartPole. 

YOUR TASK: Implement function approximation for finding the optimal value of theta.
You know a solution (given below) which you can use later to debug.
'''

def featurize(s,a):
    return (2*a-1)*s

def Q(s,a, theta):
    vec = featurize(s,a)
    return np.dot(vec,theta)

env = gym.make("CartPole-v0")
actions = range(env.action_space.n)
state = env.reset()
tot_reward = 0
theta_star = np.array([0,0,3,1])

for _ in range(200):
    action = np.argmax([Q(state,a,theta_star) for a in actions])
    state, reward, done, _ = env.step(action)
    tot_reward += reward
print("Total reward: ", tot_reward)    

