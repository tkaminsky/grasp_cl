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
def student_unspec(env_params, state, goal_ball_idx = 0):
    # Find the line between the ball and the goal
    agent_state, ball_states = state
    ball_state = ball_states[goal_ball_idx]
    ball_goal = env_params["ball_goals"][goal_ball_idx]
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

    min_l = env.game.agent_size + env.game.ball_sizes[goal_ball_idx] + 3

    # If the distance between the agent and the ball is greater than 30, move towards the ball
    if atb_length > min_l:
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
    
def student(state, curr_params):
    # Says whether each ball is in its goal
    ball_status = env.game.get_info()
    # Find the smallest index of a ball that isn't in its goal
    goal_ball_idx = np.argmin(ball_status)
    print(ball_status)
    return student_unspec(curr_params, state, goal_ball_idx)

# Action is a dictionary that looks like the following - this dictionary specifies the game environment
        # "num_balls" : Int
        # "ball_sizes" : Int array
        # "ball_inits" : float array (num_balls X 2)
        # "agent_init" : float array (2, )
        # "ball_goals" : float array (num_balls X 2)


def randomize_params():
    # Randomly cloose between 1 and 5 balls
    num_balls = random.randint(1, 5)
    # Randomly choose the size of each ball, in (5,50)
    ball_sizes = [random.randint(5, 50) for i in range(num_balls)]
    # Randomly choose the initial position of each ball, in (0, 400), ensuring that the balls don't overlap
    ball_inits = []
    for i in range(num_balls):
        overlapping = True
        while overlapping:
            overlapping = False
            pos_vote = [random.randint(ball_sizes[i], 400 - ball_sizes[i]), random.randint(ball_sizes[i], 400 - ball_sizes[i])]
            for ball in ball_inits:
                if np.linalg.norm(np.array(pos_vote) - np.array(ball)) < ball_sizes[i] + ball_sizes[ball_inits.index(ball)]:
                    overlapping = True
                    break
        ball_inits.append(pos_vote)
    # Randomly choose the initial position of the agent, in (0, 400), ensuring that the agent doesn't overlap with any balls
    overlapping = True
    agent_init = [random.randint(10, 390), random.randint(10, 390)]
    while overlapping:
        overlapping = False
        agent_init = [random.randint(10, 390), random.randint(10, 390)]
        for ball in ball_inits:
            if np.linalg.norm(np.array(agent_init) - np.array(ball)) < 10 + ball_sizes[ball_inits.index(ball)]:
                overlapping = True
                break
        
    # Randomly choose the goal position of each ball, in (0, 400), ensuring that the balls don't overlap
    ball_goals = []
    for i in range(num_balls):
        overlapping = True
        while overlapping:
            overlapping = False
            pos_vote = [random.randint(ball_sizes[i], 400 - ball_sizes[i]), random.randint(ball_sizes[i], 400 - ball_sizes[i])]
            for ball in ball_goals:
                if np.linalg.norm(np.array(pos_vote) - np.array(ball)) < ball_sizes[i] + ball_sizes[ball_goals.index(ball)]:
                    overlapping = True
                    break
        ball_goals.append(pos_vote)
    
    # Create the dictionary of parameters
    params = {
        "num_balls" : num_balls,
        "ball_sizes" : ball_sizes,
        "ball_inits" : ball_inits,
        "agent_init" : agent_init,
        "ball_goals" : ball_goals
    }

    return params            

# params = {
#     "num_balls" : 2,
#     "ball_sizes" : [50, 50],
#     "ball_inits" : [[width / 2, height / 2], [300, 300]],
#     "agent_init" : [159, 200],
#     "ball_goals" : [[60, 100], [300, 350]]
# }    

env = particlePushTeacher(student, render_mode = 'human')
obs = env.reset()

obs, reward, term, trunc, info = env.step(randomize_params())

t = 0

while True:
    # action = f(t)
    action = randomize_params()
    t += 1
    
    #action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    
    # Render the game
    env.render()
    
    if term or trunc:
        break

env.close()