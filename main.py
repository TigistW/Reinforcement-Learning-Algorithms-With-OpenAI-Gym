# main.py

import numpy as np
from multi_armed_bandit import MultiArmedBandit
from grid_world import GridWorld
from algorithms import value_iteration, extract_policy, policy_iteration, q_learning, epsilon_greedy_policy, ucb, epsilon_greedy_bandit, ucb_bandit

# Define the grid world
n, m = 4, 4
start = (0, 0)
goal = (3, 3)
obstacles = [(1, 1), (2, 2)]
grid_world = GridWorld(n, m, start, goal, obstacles)

# Run value iteration
print("Running Value Iteration for Grid World")
V = value_iteration(grid_world)
policy = extract_policy(grid_world, V)
print("Optimal Value Function:")
print(V)
print("Optimal Policy:")
print(policy)

# Run policy iteration
print("\nRunning Policy Iteration for Grid World")
policy, V = policy_iteration(grid_world)
print("Optimal Value Function:")
print(V)
print("Optimal Policy:")
print(policy)

# Run Q-learning
print("\nRunning Q-Learning for Grid World")
Q = q_learning(grid_world)
policy = np.argmax(Q, axis=2)
print("Q-Values:")
print(Q)
print("Derived Policy:")
print(policy)

# Define the multi-armed bandit
k = 10
bandit = MultiArmedBandit(k)

# Run epsilon-greedy bandit
print("\nRunning Epsilon-Greedy for Multi-Armed Bandit")
total_reward, q_estimates = epsilon_greedy_bandit(bandit, steps=1000, epsilon=0.1)
print("Total Reward:", total_reward)
print("Q-Estimates:")
print(q_estimates)

# Run UCB bandit
print("\nRunning UCB for Multi-Armed Bandit")
total_reward, q_estimates = ucb_bandit(bandit, steps=1000, c=2)
print("Total Reward:", total_reward)
print("Q-Estimates:")
print(q_estimates)
