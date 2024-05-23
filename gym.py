


import numpy as np
import random
import gymnasium as gym
from algorithms import epsilon_greedy_bandit, ucb_bandit
from multi_armed_bandit import MultiArmedBandit

# Define the environment
env = gym.make("FrozenLake-v1", render_mode="ansi")
env.reset()

def value_iteration_gym(env, gamma=0.9, theta=1e-6):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            v = V[state]
            max_value = float('-inf')
            for action in range(env.action_space.n):
                transitions = env.P[state][action]
                value = sum(prob * (reward + gamma * V[next_state])
                            for prob, next_state, reward, done in transitions)
                if value > max_value:
                    max_value = value
            V[state] = max_value
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

def extract_policy_gym(env, V, gamma=0.9):
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        max_value = float('-inf')
        best_action = 0
        for action in range(env.action_space.n):
            transitions = env.P[state][action]
            value = sum(prob * (reward + gamma * V[next_state])
                        for prob, next_state, reward, done in transitions)
            if value > max_value:
                max_value = value
                best_action = action
        policy[state] = best_action
    return policy

def policy_iteration_gym(env, gamma=0.9):
    policy = np.random.choice(env.action_space.n, size=env.observation_space.n)
    V = np.zeros(env.observation_space.n)
    is_policy_stable = False
    
    while not is_policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for state in range(env.observation_space.n):
                v = V[state]
                action = policy[state]
                transitions = env.P[state][action]
                V[state] = sum(prob * (reward + gamma * V[next_state])
                               for prob, next_state, reward, done in transitions)
                delta = max(delta, abs(v - V[state]))
            if delta < 1e-6:
                break
        
        # Policy Improvement
        is_policy_stable = True
        for state in range(env.observation_space.n):
            old_action = policy[state]
            max_value = float('-inf')
            best_action = 0
            for action in range(env.action_space.n):
                transitions = env.P[state][action]
                value = sum(prob * (reward + gamma * V[next_state])
                            for prob, next_state, reward, done in transitions)
                if value > max_value:
                    max_value = value
                    best_action = action
            policy[state] = best_action
            if old_action != policy[state]:
                is_policy_stable = False
    return policy, V

def q_learning_gym(env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _, _ = env.step(action)
            old_value = Q[state, action]
            next_max = np.max(Q[next_state])
            Q[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)
            state = next_state
    return Q

# Run value iteration
print("Running Value Iteration for FrozenLake-v1")
V = value_iteration_gym(env)
policy = extract_policy_gym(env, V)
print("Optimal Value Function:")
print(V.reshape((4, 4)))
print("Optimal Policy:")
print(policy.reshape((4, 4)))

# Run policy iteration
print("\nRunning Policy Iteration for FrozenLake-v1")
policy, V = policy_iteration_gym(env)
print("Optimal Value Function:")
print(V.reshape((4, 4)))
print("Optimal Policy:")
print(policy.reshape((4, 4)))

# Run Q-learning
print("\nRunning Q-Learning for FrozenLake-v1")
Q = q_learning_gym(env)
policy = np.argmax(Q, axis=1)
print("Q-Values:")
print(Q)
print("Derived Policy:")
print(policy.reshape((4, 4)))

