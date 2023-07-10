import pygame
import random
import math
import gym
from gym import Env
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pygame_helpers import *
from particle_push_env import particlePush

# Particle push class
class particlePushTeacher(Env):
    def __init__(self, student, render_mode='human'):
        super(particlePushTeacher, self).__init__()
        self.student = student
        self.game = None

        self.render_mode = render_mode

        self.reward = 0
        self.t = 0
        self.T = 100

    def reset(self):
        self.t = 0
        self.reward = 0
        return
    
    def render(self):
        if self.render_mode == 'rgb_array':
            return self.game.render()
        elif self.render_mode == 'human':
            # Render the game
            im = self.game.render()
            # Scale im to 2 times its size
            im = cv2.resize(im, (0, 0), fx=2, fy=2)
            cv2.imshow('Particle Push', im)
            cv2.waitKey(10)
            return self.screen
        self.game.draw_elements_on_canvas()

        return self.screen
    
    def get_agent_reward(self):
        return self.game.reward

    def get_teacher_reward(self):
        return self.reward
    
    def step(self, action):
        # Action is a dictionary that looks like the following - this dictionary specifies the game environment
        # "num_balls" : Int
        # "ball_sizes" : Int array
        # "ball_inits" : float array (num_balls X 2)
        # "agent_init" : float array (2, )
        # "ball_goals" : float array (num_balls X 2)

        # Create a new particlePush environment with the specified parameters
        # Sample 9 random initial agent locations

        self.game = particlePush(action['num_balls'], action['ball_sizes'], action['ball_inits'], action['agent_init'], action['ball_goals'], render_mode='rgb_array')

        self.screen = self.game.screen
        obs = self.game.reset()

        # Run a round of the game with the student policy
        while True:
            student_action = self.student(obs, action)
            
            
            obs, reward, term, trunc, info = self.game.step(student_action)
            
            # Render the game
            self.render()
            

            
            if term or trunc:
                break
        
        self.reward += - self.game.reward

        return self.screen, self.game.reward, False, False, {}
