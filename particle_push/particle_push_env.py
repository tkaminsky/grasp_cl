import pygame
import random
import math
import gym
from gym import Env
import numpy as np
from gym.spaces import Tuple, Discrete, Box

from pygame_helpers import *

# Particle push class
class particlePush(Env):
    def __init__(self, num_balls, ball_sizes, ball_inits, agent_init, ball_goals=None, render_mode='human'):
        super(particlePush, self).__init__()

        # Initialize render settings
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Particle Push')
        self.render_mode = render_mode

        # Specify POMDP parameters - in this case, ball number, size, and location + agent_location
        self.num_balls = num_balls
        self.ball_sizes = ball_sizes
        self.ball_inits = ball_inits
        self.ball_goals = ball_goals
        self.agent_init = agent_init

        # Static hyperparameters
        self.agent_weight = 1
        self.agent_size = 10
        self.ball_weight = 1e-6
        self.v = 10
        self.reward = 0
        self.reward_threshold = 10
        self.reward_on_success = 1
        self.reward_on_failure = 0
        self.t = 0
        self.T = 10000
        self.goal_reached = False

        self.observation_space = Tuple()

        # Initialize environment
        self.reset()

    def reset(self):
        self.balls = []
        self.agent = None
        self.reward = 0
        self.t = 0
        self.goal_reached = False
        
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

        self.clock = pygame.time.Clock()
        self.selected_particle = None
        self.running = True

        self.draw_elements_on_canvas()

        return self.get_state()
    
    def render(self):
        self.draw_elements_on_canvas()
        return self.screen
    
    def close(self):
        pygame.quit()
        quit()

    def draw_elements_on_canvas(self):
        self.screen.fill(background_colour)

        for i, particle in enumerate(self.balls):
            particle.display(self.screen)
            pygame.draw.circle(self.screen, (0,255,0), (self.ball_goals[i][0], self.ball_goals[i][1]), 20, 0)

        self.agent.display(self.screen)
        
        pygame.display.flip()

    def get_state(self):
        agent_state = np.array([self.agent.x, self.agent.y])
        ball_states = []
        for ball in self.balls:
            ball_states.append([ball.x, ball.y])
        ball_states = np.array(ball_states)
        return agent_state, ball_states

    def get_reward(self):
        dist = 0
        for i, ball in enumerate(self.balls):
            dist += np.sqrt( (ball.x - self.ball_goals[i][0])**2 + (ball.y - self.ball_goals[i][1])**2 )
        if dist < self.reward_threshold:
            return self.reward_on_success
        return self.reward_on_failure

    def step(self, action):
        # Parametrize the action ( vector with norm <= 1 )
        action = action * self.v

        dx = action[0]
        dy = action[1]

        self.agent.angle = 0.5*math.pi + math.atan2(dy, dx)
        self.agent.speed = math.hypot(dx, dy) * 0.1

        self.agent.move()
        self.agent.bounce()

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

        term = True if self.goal_reached else False
        trunc = True if self.t >= self.T else False

        self.clock.tick(200)

        return self.get_state(), curr_reward, term, trunc, {}
