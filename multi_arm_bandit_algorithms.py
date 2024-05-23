import numpy as np
import random

class MultiArmedBandit:
    def __init__(self, k):
        self.k = k
        self.probabilities = np.random.rand(k)

    def pull(self, arm):
        return 1 if np.random.rand() < self.probabilities[arm] else 0

# Value Iteration for Multi-Armed Bandit
def value_iteration_bandit(bandit, gamma=0.9, theta=1e-6, steps=1000):
    V = np.zeros(bandit.k)
    for _ in range(steps):
        delta = 0
        for state in range(bandit.k):
            v = V[state]
            reward = bandit.pull(state)
            V[state] = reward + gamma * max(V)
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    policy = np.argmax(V)
    return V, policy

# Policy Iteration for Multi-Armed Bandit
def policy_iteration_bandit(bandit, gamma=0.9, steps=1000):
    policy = np.random.choice(bandit.k)
    V = np.zeros(bandit.k)
    is_policy_stable = False
    
    while not is_policy_stable:
        # Policy Evaluation
        for _ in range(steps):
            delta = 0
            for state in range(bandit.k):
                v = V[state]
                reward = bandit.pull(state)
                V[state] = reward + gamma * V[state]
                delta = max(delta, abs(v - V[state]))
            if delta < 1e-6:
                break
        
        # Policy Improvement
        old_policy = policy
        q_values = [bandit.pull(a) + gamma * V[a] for a in range(bandit.k)]
        policy = np.argmax(q_values)
        if old_policy == policy:
            is_policy_stable = True
    return V, policy

# Q-Learning for Multi-Armed Bandit
def q_learning_bandit(bandit, alpha=0.1, gamma=0.9, epsilon=0.1, steps=1000):
    Q = np.zeros(bandit.k)
    action_counts = np.zeros(bandit.k)
    total_reward = 0
    
    for _ in range(steps):
        if random.random() < epsilon:
            action = random.randint(0, bandit.k - 1)
        else:
            action = np.argmax(Q)
        
        reward = bandit.pull(action)
        total_reward += reward
        action_counts[action] += 1
        Q[action] += alpha * (reward + gamma * np.max(Q) - Q[action])
    
    return total_reward, Q

# Epsilon-Greedy Policy for Multi-Armed Bandit
def epsilon_greedy_bandit(bandit, steps=1000, epsilon=0.1):
    k = bandit.k
    q_estimates = np.zeros(k)
    action_counts = np.zeros(k)
    total_reward = 0
    
    for _ in range(steps):
        if random.random() < epsilon:
            action = random.randint(0, k - 1)
        else:
            action = np.argmax(q_estimates)
        
        reward = bandit.pull(action)
        total_reward += reward
        action_counts[action] += 1
        q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]
    
    return total_reward, q_estimates

# UCB Algorithm for Multi-Armed Bandit
def ucb_bandit(bandit, steps=1000, c=2):
    k = bandit.k
    q_estimates = np.zeros(k)
    action_counts = np.zeros(k)
    total_reward = 0

    for t in range(1, steps + 1):
        if 0 in action_counts:
            action = np.argmin(action_counts)
        else:
            ucb_values = q_estimates + c * np.sqrt(np.log(t) / (action_counts + 1e-5))
            action = np.argmax(ucb_values)

        reward = bandit.pull(action)
        total_reward += reward
        action_counts[action] += 1
        q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]

    return total_reward, q_estimates

# Initialize the multi-armed bandit problem
k = 20
bandit = MultiArmedBandit(k)

# Run Value Iteration
print("\nRunning Value Iteration for Multi-Armed Bandit")
V, policy = value_iteration_bandit(bandit)
print("Value Function:", V)
print("Derived Policy:", policy)

# Run Policy Iteration
print("\nRunning Policy Iteration for Multi-Armed Bandit")
V, policy = policy_iteration_bandit(bandit)
print("Value Function:", V)
print("Derived Policy:", policy)

# Run Q-Learning
print("\nRunning Q-Learning for Multi-Armed Bandit")
total_reward, q_estimates = q_learning_bandit(bandit)
print("Total Reward:", total_reward)
print("Q-Estimates:")
print(q_estimates)

# Run Epsilon-Greedy Bandit
print("\nRunning Epsilon-Greedy for Multi-Armed Bandit")
total_reward, q_estimates = epsilon_greedy_bandit(bandit, steps=1000, epsilon=0.1)
print("Total Reward:", total_reward)
print("Q-Estimates:")
print(q_estimates)

# Run UCB Bandit
print("\nRunning UCB for Multi-Armed Bandit")
total_reward, q_estimates = ucb_bandit(bandit, steps=1000, c=2)
print("Total Reward:", total_reward)
print("Q-Estimates:")
print(q_estimates)
