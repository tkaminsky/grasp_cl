import pygame
from pygame import font
import random
import math
# import gymnasium as gym
from gymnasium import Env
import numpy as np
import gymnasium.spaces as spaces
from .pygame_helpers import *

# Particle push class
class particleMove(Env):
    def __init__(self, action_space_size=16, agent_init=None, agent_goal=None, render_mode='rgb_array'):
        super(particleMove, self).__init__()

        # Initialize render settings
        if render_mode == 'rgb_array':
            self.screen = pygame.display.set_mode((width, height), flags=pygame.HIDDEN)
        elif render_mode == 'human':
            self.screen = pygame.display.set_mode((width, height))
            pygame.font.init()
            self.font = font.SysFont("candara", 35) 
            
        
        pygame.display.set_caption('Particle Push')
        self.w = width
        self.h = height
        self.render_mode = render_mode

        # Specify POMDP parameters - in this case, ball number, size, and location + agent_location
        self.agent_goal = [200, 300] if agent_goal is None else agent_goal
        self.agent_init = [200, 50] if agent_init is None else agent_init

        # Static hyperparameters
        self.agent_weight = 1
        self.agent_size = 10
        self.v = 50
        self.reward = 0
        self.reward_threshold = 10
        self.reward_on_success = 1
        self.reward_on_failure = 0
        self.t = 0
        self.T = 1000
        self.goal_reached = False

        self.observation_space = spaces.Box(low=np.array([[-1.,-1.],[-1.,-1.]]), high=np.array([[1.,1.],[1.,1.]]), shape=(2,2), dtype=np.float32)

        # Action space is a set of simple 2D movements
        self.action_space = spaces.Discrete(action_space_size)
        # Uniformly sample action_space_size angles from 0 to 2pi
        self.action_map = {i: [math.cos(2*math.pi*i/action_space_size), math.sin(2*math.pi*i/action_space_size)] for i in range(action_space_size)}
        # self.action_map = {0: [0, 1], 1: [0, -1], 2: [1, 0], 3: [-1, 0]}

        # Initialize environment
        self.reset()

    # Places balls at initial locations
    def reset(self, seed=None, options=None):
        self.agent = None
        self.reward = 0
        self.t = 0
        self.goal_reached = False

        # Randomize agent_init
        self.agent_init = np.random.uniform(low=10, high=(width - 10), size=(2,))

        # Make the agent
        self.agent = Particle(self.agent_init, self.agent_size, self.agent_weight, name="Agent")
        self.agent.colour = (255, 0, 0)
        self.agent.speed = 0
        self.agent.angle = 0
        
        # Initialize pygame
        self.clock = pygame.time.Clock()
        self.selected_particle = None
        self.running = True

        self.render()

        return self.get_state(), {}
    
    # Wrapper for draw_elements_on_canvas
    def render(self):
        if self.render_mode != 'None':
            vis = self.draw_elements_on_canvas()
            return vis
    
    # End the game
    def close(self):
        pygame.quit()
        quit()

    # Draws the current state of the game on the canvas
    def draw_elements_on_canvas(self):
        self.screen.fill(background_colour)

        pygame.draw.circle(self.screen, (0,255,0), (self.agent_goal[0], self.agent_goal[1]), 5, 0)
        self.agent.display(self.screen)

        # Render the pygame display window
        if self.render_mode == 'human':
            pygame.display.flip()
            return None
        # Return the rendered image as a numpy array
        elif self.render_mode == 'rgb_array':
            x3 = pygame.surfarray.array3d(self.screen)
            return np.uint8(x3)

    # Returns the current state of the game
    def get_state(self):

        #return agent_state, ball_states as a 2x2 array
        state = np.array([[self.agent.x, self.agent.y], [self.agent_goal[0], self.agent_goal[1]]], dtype=np.float32)
        # Subtract self.w/2 and self.h/2 from each element of state
        state = state - np.array([self.w/2, self.h/2])
        # Divide each element of state by self.w/2 and self.h/2
        state = state / np.array([self.w/2, self.h/2])
        return state

    # Returns the reward for the current state of the game
    def get_reward(self):
        info = self.get_info()
        if info:
            self.goal_reached = True
            return self.reward_on_success
        return self.reward_on_failure
    
    # Returns whether each ball is on its goal
    def get_info(self):
        on_goal = False
        dist = np.sqrt( (self.agent.x - self.agent_goal[0])**2 + (self.agent.y - self.agent_goal[1])**2 )
        if dist < self.reward_threshold:
            on_goal = True
        return on_goal
    
    # Returns the parameters of the game
    def get_params(self):
        # Params is a dictionary that looks like the following - this dictionary specifies the game environment
        # "num_balls" : Int
        # "ball_sizes" : Int array
        # "ball_inits" : float array (num_balls X 2)
        # "agent_init" : float array (2, )
        # "ball_goals" : float array (num_balls X 2)
        params = {
            "agent_init" : self.agent_init,
            "agent_goal" : self.agent_goal
        }
        return params
    
    def _dense_reward_helper(self, init_dist, curr_dist, m, min = None):
        # print(init_dist, curr_dist)
        if curr_dist < init_dist or min is None:
            return m * (init_dist - curr_dist) / init_dist
        return 0
    
    def dense_reward(self):
        reward = 0
        # Add a small reward if the agent is closer to a ball than it was at initialization
        init_dist = np.sqrt( (self.agent_init[0] - self.agent_goal[0])**2 + (self.agent_init[1] - self.agent_goal[1])**2 )
        curr_dist = np.sqrt( (self.agent.x - self.agent_goal[0])**2 + (self.agent.y - self.agent_goal[1])**2 )
        # print(f"init_dist: {init_dist}, curr_dist: {curr_dist}")
        reward  += self._dense_reward_helper(init_dist, curr_dist, 1)

        return reward

            
    # Runs one step of the game
    def step(self, action):
        # Add the action to the display as text
            # self.screen.blit(self.font.render(str(action), True, (0,0,0)), (0,0))
        # Parametrize the action ( vector with norm <= 1 )
        # Scale the action to be a unit vector if it is too large
        action_arr = self.action_map[action]
        dx = action_arr[0] * self.v
        dy = action_arr[1] * self.v

        # Print the norm of the action arr
        # print(f"Norm of action_arr: {np.linalg.norm(np.array([dx, dy]))}")

        self.agent.angle = 0.5*math.pi + math.atan2(dy, dx)
        self.agent.speed = math.hypot(dx, dy) * 0.1

        self.agent.move()
        self.agent.bounce()

        self.agent.angle = 0.5*math.pi + math.atan2(dy, dx)
        self.agent.speed = math.hypot(dx, dy) * 0.1

        # Get the reward
        curr_reward = self.get_reward()
        self.reward += curr_reward

        # Check if either end condition is met
        term = True if self.goal_reached else False
        trunc = True if self.t >= self.T else False
        done = term or trunc

        if self.render_mode != 'None':
            pygame.display.set_caption(str(self.get_state()))

        self.clock.tick(200)

        self.t += 1

        reward = self.dense_reward() + curr_reward * 10_000

        # return self.get_state(), self.dense_reward(), term, trunc, {}
        return self.get_state(), reward, term, trunc, {}
