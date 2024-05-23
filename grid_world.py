# grid_world.py

import numpy as np

class GridWorld:
    def __init__(self, n, m, start, goal, obstacles):
        self.n = n
        self.m = m
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles)
        self.grid = np.zeros((n, m))
        for obs in obstacles:
            self.grid[obs] = -10
        self.grid[goal] = 10
        self.actions = ['up', 'down', 'left', 'right']
    
    def step(self, state, action):
        i, j = state
        if action == 'up':
            next_state = (max(0, i-1), j)
        elif action == 'down':
            next_state = (min(self.n-1, i+1), j)
        elif action == 'left':
            next_state = (i, max(0, j-1))
        elif action == 'right':
            next_state = (i, min(self.m-1, j+1))
        
        if next_state in self.obstacles:
            return state, -10, False
        if next_state == self.goal:
            return next_state, 10, True
        return next_state, -1, False
    
    def reset(self):
        return self.start
