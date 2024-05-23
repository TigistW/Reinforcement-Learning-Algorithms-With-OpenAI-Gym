import numpy as np
import random
import gymnasium as gym


# Define the grid_worldironment
grid_world = gym.make("FrozenLake-v1", render_mode="ansi")
grid_world.reset()

def value_iteration_gym(grid_world, gamma=0.9, theta=1e-6):
    V = np.zeros(grid_world.observation_space.n)
    while True:
        delta = 0
        for state in range(grid_world.observation_space.n):
            v = V[state]
            max_value = float('-inf')
            for action in range(grid_world.action_space.n):
                transitions = grid_world.P[state][action]
                value = sum(prob * (reward + gamma * V[next_state])
                            for prob, next_state, reward, done in transitions)
                if value > max_value:
                    max_value = value
            V[state] = max_value
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

def extract_policy_gym(grid_world, V, gamma=0.9):
    policy = np.zeros(grid_world.observation_space.n, dtype=int)
    for state in range(grid_world.observation_space.n):
        max_value = float('-inf')
        best_action = 0
        for action in range(grid_world.action_space.n):
            transitions = grid_world.P[state][action]
            value = sum(prob * (reward + gamma * V[next_state])
                        for prob, next_state, reward, done in transitions)
            if value > max_value:
                max_value = value
                best_action = action
        policy[state] = best_action
    return policy

def policy_iteration_gym(grid_world, gamma=0.9):
    policy = np.random.choice(grid_world.action_space.n, size=grid_world.observation_space.n)
    V = np.zeros(grid_world.observation_space.n)
    is_policy_stable = False
    
    while not is_policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for state in range(grid_world.observation_space.n):
                v = V[state]
                action = policy[state]
                transitions = grid_world.P[state][action]
                V[state] = sum(prob * (reward + gamma * V[next_state])
                               for prob, next_state, reward, done in transitions)
                delta = max(delta, abs(v - V[state]))
            if delta < 1e-6:
                break
        
        # Policy Improvement
        is_policy_stable = True
        for state in range(grid_world.observation_space.n):
            old_action = policy[state]
            max_value = float('-inf')
            best_action = 0
            for action in range(grid_world.action_space.n):
                transitions = grid_world.P[state][action]
                value = sum(prob * (reward + gamma * V[next_state])
                            for prob, next_state, reward, done in transitions)
                if value > max_value:
                    max_value = value
                    best_action = action
            policy[state] = best_action
            if old_action != policy[state]:
                is_policy_stable = False
    return policy, V

def q_learning_gym(grid_world, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    Q = np.zeros((grid_world.observation_space.n, grid_world.action_space.n))
    for _ in range(episodes):
        state, _ = grid_world.reset()
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = grid_world.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _, _ = grid_world.step(action)
            old_value = Q[state, action]
            next_max = np.max(Q[next_state])
            Q[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)
            state = next_state
    return Q


# Function to perform Epsilon-Greedy Policy on the Gym grid_worldironment
def epsilon_greedy_gym(grid_world, Q, epsilon=0.1, episodes=1000):
    policy = np.zeros(grid_world.observation_space.n, dtype=int)  # Initialize policy to zeros
    for _ in range(episodes):
        state, _ = grid_world.reset()  # Reset the grid_worldironment to start a new episode
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = grid_world.action_space.sample()  # Exploration: choose a random action
            else:
                action = np.argmax(Q[state])  # Exploitation: choose the best known action
            
            next_state, reward, done, _, _ = grid_world.step(action)  # Take the action and observe the outcome
            state = next_state  # Move to the next state
        policy[state] = action
    return policy



# Function to perform UCB Algorithm on the Gym grid_worldironment
def ucb_gym(grid_world, c=2, episodes=1000, gamma = 0.9):
    Q = np.zeros((grid_world.observation_space.n, grid_world.action_space.n))  # Initialize Q-table to zeros
    counts = np.zeros((grid_world.observation_space.n, grid_world.action_space.n))  # Initialize counts to zeros
    for _ in range(episodes):
        state, _ = grid_world.reset()  # Reset the grid_worldironment to start a new episode
        done = False
        t = 0
        while not done:
            total_counts = np.sum(counts[state])
            if total_counts == 0:
                action = grid_world.action_space.sample()
            else:
                ucb_values = Q[state] + c * np.sqrt(np.log(t + 1) / (counts[state] + 1))
                action = np.argmax(ucb_values)
            
            next_state, reward, done, _, _ = grid_world.step(action)  # Take the action and observe the outcome
            counts[state, action] += 1
            old_value = Q[state, action]
            next_max = np.max(Q[next_state])
            Q[state, action] = old_value + (reward + gamma * next_max - old_value) / counts[state, action]
            state = next_state
            t += 1
    policy = np.argmax(Q, axis=1)
    return policy

# Run value iteration
print("Running Value Iteration for FrozenLake-v1")
V = value_iteration_gym(grid_world)
policy = extract_policy_gym(grid_world, V)
print("Optimal Value Function:")
print(V.reshape((4, 4)))
print("Optimal Policy:")
print(policy.reshape((4, 4)))

# Run policy iteration
print("\nRunning Policy Iteration for FrozenLake-v1")
policy, V = policy_iteration_gym(grid_world)
print("Optimal Value Function:")
print(V.reshape((4, 4)))
print("Optimal Policy:")
print(policy.reshape((4, 4)))

# Run Q-learning
print("\nRunning Q-Learning for FrozenLake-v1")
Q = q_learning_gym(grid_world)
policy = np.argmax(Q, axis=1)
print("Q-Values:")
print(Q)
print("Derived Policy:")
print(policy.reshape((4, 4)))

# Run Epsilon-Greedy Policy
print("\nRunning Epsilon-Greedy Policy for FrozenLake-v1")
policy = epsilon_greedy_gym(grid_world, Q)
print("Epsilon-Greedy Policy:")
print(policy.reshape((4, 4)))

# Run UCB
print("\nRunning UCB for FrozenLake-v1")
policy = ucb_gym(grid_world)
print("UCB Policy:")
print(policy.reshape((4,4)))