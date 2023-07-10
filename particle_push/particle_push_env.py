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
class particlePush(Env):
    def __init__(self, num_balls=None, ball_sizes=None, ball_inits=None, agent_init=None, ball_goals=None, render_mode='None'):
        super(particlePush, self).__init__()

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
        self.num_balls = num_balls
        self.ball_sizes = ball_sizes
        self.ball_inits = ball_inits
        self.ball_goals = ball_goals
        self.agent_init = agent_init

        # Choose which fields are randomized
        self.randomize_balls = True if self.num_balls is None else False
        self.randomize_ball_sizes = True if self.ball_sizes is None else False
        self.randomize_ball_inits = True if self.ball_inits is None else False
        self.randomize_ball_goals = True if self.ball_goals is None else False
        self.randomize_agent_init = True if self.agent_init is None else False

        # Static hyperparameters
        self.agent_weight = 1
        self.agent_size = 10
        self.ball_weight = 1e-6
        self.v = 20
        self.reward = 0
        self.reward_threshold = 5
        self.reward_on_success = 1
        self.reward_on_failure = 0
        self.t = 0
        self.T = 500
        self.goal_reached = False

        # Observation space is a tuple of agent state and ball states
        if self.randomize_balls is True:
            self.observation_space = None
        else:
            # print("Num balls: ", self.num_balls)
            # self.observation_space = spaces.Dict({
            #     'agent_state': spaces.Box(low=np.array([0.,0.]), high=np.array([self.w, self.h]), shape=(2,), dtype=np.float32),
            #     'ball_states': spaces.Box(low=np.array([[0.,0.]]*self.num_balls), high=np.array([[self.w, self.h]]*self.num_balls), shape=(self.num_balls, 2), dtype=np.float32)
            # })
            # self.observation_space = spaces.Dict({
            #     'agent_state': spaces.Box(low=np.array([0.,0.]), high=np.array([self.w, self.h]), shape=(2,), dtype=np.float32),
            #     'ball_states': spaces.Box(low=np.array([[0.,0.], [0.,0.]]), high=np.array([[self.w, self.h], [self.w, self.h]]), shape=(2, 2), dtype=np.float32)
            # })
            # self.observation_space = spaces.Dict({
            #     'agent_state': spaces.Box(low=np.array([0.,0.]), high=np.array([self.w, self.h]), shape=(2,), dtype=np.float32),
            #     'ball_states': spaces.Box(low=np.array([[0.,0.]]), high=np.array([[self.w, self.h]]), shape=(1, 2), dtype=np.float32)
            # })
            # self.observation_space = spaces.Box(low=np.array([[0.,0.], [0.,0.]]), high=np.array([[self.w, self.h], [self.w, self.h]]), shape=(2,2), dtype=np.float32)
            self.observation_space = spaces.Box(low=np.array([[-1.,-1.],[-1.,-1.],[-1.,-1.]]), high=np.array([[1.,1.],[1.,1.],[1.,1.]]), shape=(3,2), dtype=np.float32)

        # Action space is a tuple of x and y directions, which are scaled to lie on the unit circle
        self.action_space = spaces.Box(low=np.array([-1.,-1.]), high=np.array([1.,1.]), shape=(2,), dtype=np.float32)

        # Initialize environment
        self.reset()

    # Places balls at initial locations
    def reset(self, seed=None, options=None):
        self.balls = []
        self.agent = None
        self.reward = 0
        self.t = 0
        self.goal_reached = False

        # Randomize every unspecified parameter
        self.num_balls = 1
        self.randomize_balls = False
        self.randomize()       

        
        # Make the balls
        for n in range(self.num_balls):
            particle = Particle(self.ball_inits[n], self.ball_sizes[n], self.ball_weight)
            particle.colour = (100, 100, 255)
            particle.speed = 0
            particle.angle = 0
            self.balls.append(particle)

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

        return self.get_state()
    
    # Randomizes unspecified parameters
    def randomize(self):
         # Choose a random number of balls between 1 and 5
        if self.randomize_balls:
            self.num_balls = random.randint(1, 5)
            self.observation_space = spaces.Dict({
                'agent_state': spaces.Box(low=np.array([0.,0.]), high=np.array([self.w, self.h]), shape=(2,), dtype=np.float32),
                'ball_states': spaces.Box(low=np.array([[0.,0.]]*self.num_balls), high=np.array([[self.w, self.h]]*self.num_balls), shape=(self.num_balls, 2), dtype=np.float32)
            })
        # Choose a random ball size between 10 and 50
        if self.randomize_ball_sizes:
            self.ball_sizes = [random.randint(10, 50) for _ in range(self.num_balls)]

        # Choose random ball inits, making sure they don't overlap
        if self.randomize_ball_inits:
            self.ball_inits = [[random.randint(self.ball_sizes[i], self.w - self.ball_sizes[i]), random.randint(self.ball_sizes[i], self.h - self.ball_sizes[i])] for i in range(self.num_balls)]
            for i in range(self.num_balls):
                for j in range(i):
                    while np.linalg.norm(np.array(self.ball_inits[i]) - np.array(self.ball_inits[j])) < self.ball_sizes[i] + self.ball_sizes[j]:
                        self.ball_inits[i] = [random.randint(self.ball_sizes[i], self.w - self.ball_sizes[i]), random.randint(self.ball_sizes[i], self.h - self.ball_sizes[i])]
        # Choose random ball goals, making sure they don't overlap
        if self.randomize_ball_goals:
            self.ball_goals = [[random.randint(self.ball_sizes[i], self.w - self.ball_sizes[i]), random.randint(self.ball_sizes[i], self.h - self.ball_sizes[i])] for i in range(self.num_balls)]
            for i in range(self.num_balls):
                for j in range(i):
                    while np.linalg.norm(np.array(self.ball_goals[i]) - np.array(self.ball_goals[j])) < self.ball_sizes[i] + self.ball_sizes[j]:
                        self.ball_goals[i] = [random.randint(self.ball_sizes[i], self.w - self.ball_sizes[i]), random.randint(self.ball_sizes[i], self.h - self.ball_sizes[i])]
        # Choose random agent init, making sure it doesn't overlap with any balls
        if self.randomize_agent_init:
            self.agent_init = [random.randint(self.agent_size, self.w - self.agent_size), random.randint(self.agent_size, self.h - self.agent_size)]
            # Make sure the agent doesn't spawn on top of a ball
            for i in range(self.num_balls):
                while np.linalg.norm(np.array(self.agent_init) - np.array(self.ball_inits[i])) < self.ball_sizes[i] + self.agent_size:
                    self.agent_init = [random.randint(self.agent_size, self.w - self.agent_size), random.randint(self.agent_size, self.h - self.agent_size)]
    
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

        for i, particle in enumerate(self.balls):
            particle.display(self.screen)
            pygame.draw.circle(self.screen, (0,255,0), (self.ball_goals[i][0], self.ball_goals[i][1]), 5, 0)
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
        agent_state = np.array([self.agent.x, self.agent.y])
        ball_states = []
        for ball in self.balls:
            ball_states.append([ball.x, ball.y])
        ball_states = np.array(ball_states)

        # state = {
        #     'agent_state': agent_state,
        #     'ball_states': ball_states
        # }
        #return agent_state, ball_states as a 2x2 array
        # state = np.concatenate((agent_state, ball_states[0]), axis=0)
        state = np.array([[self.agent.x, self.agent.y],[self.balls[0].x, self.balls[0].y], [self.ball_goals[0][0], self.ball_goals[0][1]]], dtype=np.float32)
        # Subtract self.w/2 and self.h/2 from each element of state
        state = state - np.array([self.w/2, self.h/2])
        # Divide each element of state by self.w/2 and self.h/2
        state = state / np.array([self.w/2, self.h/2])
        return state

    # Returns the reward for the current state of the game
    def get_reward(self):
        info = self.get_info()
        if all(info):
            self.goal_reached = True
            return self.reward_on_success
        return self.reward_on_failure
    
    # Returns whether each ball is on its goal
    def get_info(self):
        on_goal = [False for _ in range(self.num_balls)]
        for i, ball in enumerate(self.balls):
            dist = np.sqrt( (ball.x - self.ball_goals[i][0])**2 + (ball.y - self.ball_goals[i][1])**2 )
            if dist < self.reward_threshold:
                on_goal[i] = True
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
            "num_balls" : self.num_balls,
            "ball_sizes" : self.ball_sizes,
            "ball_inits" : self.ball_inits,
            "agent_init" : self.agent_init,
            "ball_goals" : self.ball_goals
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
        for i, ball in enumerate(self.balls):
            init_dist = np.sqrt( (self.agent_init[0] - ball.x)**2 + (ball.y - self.agent_init[1])**2 )
            curr_dist = np.sqrt( (ball.x - self.agent.x)**2 + (ball.y - self.agent.y)**2 )
            reward  += self._dense_reward_helper(init_dist, curr_dist, 1)
        # Add a large reward if the ball is closer to its goal than it was at initialization
        for i, ball in enumerate(self.balls):
            init_dist = np.sqrt( (self.ball_inits[i][0] - self.ball_goals[i][0])**2 + (self.ball_inits[i][1] - self.ball_goals[i][1])**2 )
            curr_dist = np.sqrt( (ball.x - self.ball_goals[i][0])**2 + (ball.y - self.ball_goals[i][1])**2 )
            reward += self._dense_reward_helper(init_dist, curr_dist, 100)
        # print(reward)
        # print(self.t)
        # Check if either end condition is met
        term = True if self.goal_reached else False
        trunc = True if self.t >= self.T else False

        return reward

            
    # Runs one step of the game
    def step(self, action):
        # Add the action to the display as text
            # self.screen.blit(self.font.render(str(action), True, (0,0,0)), (0,0))
        # Parametrize the action ( vector with norm <= 1 )
        # Scale the action to be a unit vector if it is too large
        if np.linalg.norm(action) > 1:
            action = action / np.linalg.norm(action)
        
        # Scale the action by a velocity hyperparameter
        action = action * self.v

        dx = action[0]
        dy = action[1]

        self.agent.angle = 0.5*math.pi + math.atan2(dy, dx)
        self.agent.speed = math.hypot(dx, dy) * 0.1

        self.agent.move()
        self.agent.bounce()

        # Added so that the agent can still propagate collisions with the balls
        self.agent.angle = 0.5*math.pi + math.atan2(dy, dx)
        self.agent.speed = math.hypot(dx, dy) * 0.1

        # Model ball collisions
        for particle in self.balls:
            collide(self.agent, particle)
        for i, particle in enumerate(self.balls):
            particle.move()
            particle.bounce()
            for particle2 in self.balls[i+1:]:
                collide(particle, particle2)

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

        # return self.get_state(), self.dense_reward(), term, trunc, {}
        return self.get_state(), self.dense_reward(), done, {}
