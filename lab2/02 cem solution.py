# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:20:19 2018

@author: jpmaldonado
"""

import numpy as np

# the function we want to optimize for sanity check
def f(theta):
  # Here would go the evaluation of the episode
  reward = -np.sum(np.square(solution - theta))
  return reward

solution = np.array([1, 1, -0.5])


#################################
# STARTER CODE - CEM
#################################
dim_theta = 3
theta_mean = np.zeros(dim_theta)
theta_std = np.ones(dim_theta)
batch_size = 25 # number of samples per batch
elite_frac = 0.2 # fraction of samples used as elite set


n_iter = 300
# Now, for the algorithms
for _ in range(n_iter):

    # Sample parameter vectors    

    thetas = np.random.normal(theta_mean, theta_std, (batch_size,dim_theta))
    rewards = [f(theta) for theta in thetas]

    # Get elite parameters
    n_elite = int(batch_size * elite_frac)
    elite_inds = np.argsort(rewards)[batch_size - n_elite:batch_size]
    elite_thetas = [thetas[i] for i in elite_inds]
    
    # Update theta_mean, theta_std
    
    theta_mean = np.mean(elite_thetas,axis=0)
    theta_std = np.std(elite_thetas,axis=0)

print("CEM solution after {}: {}".format(n_iter, theta_mean))
#################################
# STARTER CODE - NES
#################################
theta0 = np.random.randn(3) #initial guess
npop=50
n_iter=300
sigma=0.1
alpha=0.001


theta = theta0 
for _ in range(n_iter):
    # Sample vectors from a normal distribution
    noise = np.random.randn(npop, theta.shape[0]) 
    
    # Sample function values by evaluating on the population
    rewards = np.zeros(npop)
    for i in range(npop):
        theta_i = theta + sigma*noise[i] 
        rewards[i] = f(theta_i) 
    
    # Optional: standardize (substract mean and divide by std)
    rewards = (rewards-np.mean(rewards))/np.std(rewards)
    
    # "Gradient" update
    theta = theta + alpha / (npop*sigma)*sum((rewards-f(theta)))
print("NES solution", theta)    