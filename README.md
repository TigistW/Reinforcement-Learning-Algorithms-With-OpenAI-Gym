# Reinforcement Learning Algorithms with Gymnasium

This project implements several reinforcement learning algorithms and applies them to two different environments: a grid world environment and a multi-armed bandit problem. The project uses Gymnasium's `FrozenLake-v1` environment as a grid world example.

## Table of Contents

- [Introduction](#introduction)
- [Environments](#environments)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Introduction

This project demonstrates the implementation of the following reinforcement learning algorithms:
- Value Iteration
- Policy Iteration
- Q-Learning
- Epsilon-Greedy Policy
- Upper Confidence Bound (UCB) Algorithm

## Environments

### Grid World Environment
A 2D grid where each cell can be empty, contain an obstacle, or be the goal. The agent navigates the grid to reach the goal while avoiding obstacles.

### Single-State Multi-Armed Bandit Problem
A single-state environment with multiple actions (arms). The agent tries to maximize the total reward by selecting the best arms to pull, balancing exploration and exploitation.

### Gymnasium's `FrozenLake-v1` Environment
A grid world environment where the agent navigates a frozen lake to reach a goal while avoiding holes.

## Algorithms

### Value Iteration
An algorithm that iteratively updates the value of each state based on the expected rewards.

### Policy Iteration
An algorithm that alternates between policy evaluation and policy improvement to find the optimal policy.

### Q-Learning
A model-free algorithm that learns the value of taking a particular action in a particular state.

### Epsilon-Greedy Policy
A policy that balances exploration and exploitation by selecting random actions with probability epsilon and the best-known action with probability 1-epsilon.

### Upper Confidence Bound (UCB) Algorithm
An algorithm that selects actions based on an optimism in the face of uncertainty principle, balancing exploration and exploitation.

## Project Structure

reinforcement_learning/
│
├── grid_world.py
├── multi_armed_bandit.py
├── algorithms.py
├── main.py
├── gym.py
└── requirements.txt

- `grid_world.py`: Defines the GridWorld environment.
- `multi_armed_bandit.py`: Defines the Multi-Armed Bandit environment.
- `algorithms.py`: Contains the implementation of the reinforcement learning algorithms.
- `main.py`: The entry point for running the algorithms on the environments.
- `gym.py`: The entry point for running the algorithms on the environments.
- `requirements.txt`: Lists the required Python packages.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/reinforcement-learning-algorithms.git
   cd reinforcement-learning-algorithms
2. Install the necessary packages:

`pip install -r requirements.txt`

## Usage
To run the algorithms on the environments, execute the main script:

`python main.py`
`python gym.py`

## Results

The script will output the results of running value iteration, policy iteration, and Q-learning on the `FrozenLake-v1` environment, as well as the results of running epsilon-greedy and UCB algorithms on the multi-armed bandit problem.
