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
from particle_teacher_env import particlePushTeacher

# This student tries to push the ball through a predetermined movement
def student_unspec(env_params, state):
    # Find the line between the ball and the goal
    agent_state, ball_states = state
    ball_state = ball_states[0]
    ball_goal = env_params["ball_goals"][0]
    ball_goal = np.array(ball_goal)
    ball_state = np.array(ball_state)
    agent_state = np.array(agent_state)

    # Find the vector from the ball to the goal
    goal_to_ball = ball_state - ball_goal
    btg_length = np.linalg.norm(goal_to_ball)
    # Scale the vector to have length 1
    goal_to_ball = goal_to_ball / btg_length

    # Find the vector from the agent to the ball
    ball_to_agent = agent_state - ball_state
    atb_length = np.linalg.norm(ball_to_agent)
    # Scale the vector to have length 1
    ball_to_agent = ball_to_agent / atb_length

    atg_length = np.linalg.norm(ball_goal - agent_state)

    # If the distance between the agent and the ball is greater than 30, move towards the ball
    if atb_length > 50:
        action = -ball_to_agent

    # # If the dot product of the two vectors isn't 1
    elif np.dot(goal_to_ball, ball_to_agent) < .98:
        tangent = np.array([-ball_to_agent[1], ball_to_agent[0]])
        agent_state_moved = agent_state + tangent
        new_dot_pos = np.dot(goal_to_ball, (agent_state_moved - ball_state) / np.linalg.norm(agent_state_moved - ball_state))
        agent_state_moved_neg = agent_state - tangent
        new_dot_neg = np.dot(goal_to_ball, (agent_state_moved_neg - ball_state) / np.linalg.norm(agent_state_moved_neg - ball_state))
        # Move the agent in the direction with the greater dot_product
        if new_dot_pos > new_dot_neg:
            action = tangent
        else:
            action = -tangent
    else:
        action = -ball_to_agent
    
    return action
    
def student(state):
    return student_unspec(params, state)

# Action is a dictionary that looks like the following - this dictionary specifies the game environment
        # "num_balls" : Int
        # "ball_sizes" : Int array
        # "ball_inits" : float array (num_balls X 2)
        # "agent_init" : float array (2, )
        # "ball_goals" : float array (num_balls X 2)

params = {
    "num_balls" : 1,
    "ball_sizes" : [20],
    "ball_inits" : [[width / 2, height / 2]],
    "agent_init" : [159, 200],
    "ball_goals" : [[60, 100]]
}    

env = particlePushTeacher(student, render_mode = 'human')
obs = env.reset()

obs, reward, term, trunc, info = env.step(params)

# t = 0

# while True:
#     # action = f(t)
#     action = choose_action()
#     t += 1
    
#     #action = env.action_space.sample()
#     obs, reward, term, trunc, info = env.step(action)
    
#     # Render the game
#     env.render()
    
#     if term or trunc:
#         break

# env.close()