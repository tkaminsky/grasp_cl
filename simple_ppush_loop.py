import gym
from particle_push import particlePush
import numpy as np

N = 8
# Discretize the unit circle into N directions
actions = np.array([[np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N)] for i in range(N)])