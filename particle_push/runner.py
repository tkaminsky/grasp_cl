import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time
from pygame_helpers import *
from env import particlePush

# from IPython import display

goal = [width / 2, height / 2]

def choose_action():
    # Find the direction closest to the goal
    # 1 up, 3 left, 0 down, 2 right
    # 4 stay
    x,y = env.agent.x, env.agent.y
    current_pos = current_pos = [x, y]

    # If the agent is within 5 pixels of the goal, stay
    if abs(current_pos[0] - goal[0]) < 5 and abs(current_pos[1] - goal[1]) < 5:
        # Choose a new random goal
        real_goal = False
        while not real_goal:
            goal[0] = random.randint(env.agent_size, width - env.agent_size)
            goal[1] = random.randint(env.agent_size, height - env.agent_size)
            # If under 200 pixels from center, choose a new goal
            if np.linalg.norm(np.array(goal) - np.array([width / 2, height / 2])) < 200:
                real_goal = True
                env.goal = goal
        return np.zeros(2)

    pos_dir = [goal[0] - current_pos[0], goal[1] - current_pos[1]]

    # Scale the direction vector to have length 1
    pos_dir = np.array(pos_dir)
    pos_dir = pos_dir / np.linalg.norm(pos_dir)
    # Return pos_dir as a list
    return pos_dir
    

env = particlePush(number_of_particles=10, render_mode = 'human')
obs = env.reset()

t = 0

while True:
    # action = f(t)
    action = choose_action()
    t += 1
    
    #action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    
    # Render the game
    env.render()
    
    if term or trunc:
        break

env.close()