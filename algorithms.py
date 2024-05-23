# algorithms.py

import numpy as np
import random

# Value Iteration
def value_iteration(grid_world, gamma=0.9, theta=1e-6):

    V = np.zeros((grid_world.n, grid_world.m))
    while True:
        delta = 0
        for i in range(grid_world.n):
            for j in range(grid_world.m):
                if (i, j) in grid_world.obstacles or (i, j) == grid_world.goal:
                    continue
                v = V[i, j]
                
                max_value = float('-inf')
                for action in grid_world.actions:
                    next_state, reward, prob = grid_world.step((i, j), action)
                    value = prob * (reward + gamma * V[next_state])
                    if value > max_value:
                        max_value = value
                V[i, j] = max_value
                
                delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break
    return V

def extract_policy(env, V, gamma=0.9):
    policy = np.zeros((env.n, env.m), dtype=int)
    for i in range(env.n):
        for j in range(env.m):
            if (i, j) in env.obstacles or (i, j) == env.goal:
                continue
            max_value = float('-inf')
            for action in env.actions:
                next_state, reward, prob = env.step((i, j), action)
                value = prob * (reward + gamma * V[next_state])
                if value > max_value:
                    max_value = value
                    best_action = env.actions.index(action)
            policy[i, j] = best_action
            
    return policy

# Policy Iteration
def policy_iteration(env, gamma=0.9):
    policy = np.random.choice(len(env.actions), size=(env.n, env.m))
    V = np.zeros((env.n, env.m))
    is_policy_stable = False
    
    while not is_policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for i in range(env.n):
                for j in range(env.m):
                    if (i, j) in env.obstacles or (i, j) == env.goal:
                        continue
                    v = V[i, j]
                    action = env.actions[policy[i, j]]
                    next_state, reward, prob = env.step((i, j), action)
                    V[i, j] = prob * (reward + gamma * V[next_state])
                    delta = max(delta, abs(v - V[i, j]))
            if delta < 1e-6:
                break
        
        # Policy Improvement
        is_policy_stable = True
        for i in range(env.n):
            for j in range(env.m):
                if (i, j) in env.obstacles or (i, j) == env.goal:
                    continue
                old_action = policy[i, j]
                best_action = None
                max_value = float('-inf')
                for action in env.actions:
                    next_state, reward, prob = env.step((i, j), action)
                    value = prob * (reward + gamma * V[next_state])
                    if value > max_value:
                        max_value = value
                        best_action = env.actions.index(action)
                policy[i, j] = best_action
                if old_action != policy[i, j]:
                    is_policy_stable = False
    return policy, V

# Q-Learning
def q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    Q = np.zeros((env.n, env.m, len(env.actions)))
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.actions)
            else:
                action = env.actions[np.argmax(Q[state[0], state[1]])]
            
            next_state, reward, done = env.step(state, action)
            old_value = Q[state[0], state[1], env.actions.index(action)]
            next_max = np.max(Q[next_state[0], next_state[1]])
            Q[state[0], state[1], env.actions.index(action)] = old_value + alpha * (reward + gamma * next_max - old_value)
            state = next_state
    return Q

# Epsilon-Greedy Policy
def epsilon_greedy_policy(Q, epsilon=0.1):
    def policy_fn(state):
        if random.uniform(0, 1) < epsilon:
            return random.choice(range(len(Q[state[0], state[1]])))
        return np.argmax(Q[state[0], state[1]])
    return policy_fn

# UCB Algorithm
def ucb(Q, state, c, counts, t):
    total_counts = np.sum(counts[state])
    if total_counts == 0:
        return np.random.choice(len(Q[state[0], state[1]]))
    ucb_values = Q[state[0], state[1]] + c * np.sqrt(np.log(t + 1) / (counts[state] + 1))
    return np.argmax(ucb_values)

# Epsilon-Greedy for Bandit
def epsilon_greedy_bandit(bandit, steps, epsilon=0.1):
    q_estimates = np.zeros(bandit.k)
    action_counts = np.zeros(bandit.k)
    total_reward = 0
    for step in range(steps):
        if np.random.rand() < epsilon:
            action = np.random.choice(bandit.k)
        else:
            action = np.argmax(q_estimates)
        reward = bandit.step(action)
        action_counts[action] += 1
        q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]
        total_reward += reward
    return total_reward, q_estimates

# UCB for Bandit
def ucb_bandit(bandit, steps, c):
    q_estimates = np.zeros(bandit.k)
    action_counts = np.zeros(bandit.k)
    total_reward = 0
    for step in range(steps):
        ucb_values = q_estimates + c * np.sqrt(np.log(step + 1) / (action_counts + 1))
        action = np.argmax(ucb_values)
        reward = bandit.step(action)
        action_counts[action] += 1
        q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]
        total_reward += reward
    return total_reward, q_estimates
