# multi_armed_bandit.py

import numpy as np

class MultiArmedBandit:
    def __init__(self, k):
        self.k = k
        self.q_star = np.random.normal(0, 1, k)
    
    def step(self, action):
        reward = np.random.normal(self.q_star[action], 1)
        return reward

    def reset(self):
        self.q_star = np.random.normal(0, 1, self.k)
