import pygame
import random
import math
import gym
from gym import Env
import numpy as np

from pygame_helpers import *

class particlePush(Env):
    def __init__(self, number_of_particles=5, render_mode='human'):
        super(particlePush, self).__init__()

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Particle Push')
        self.render_mode = render_mode

        self.number_of_particles = number_of_particles
        self.goal = [width / 2, height / 2]

        self.agent_density = 10000
        self.ball_density = 1

        self.v = 10
        self.agent_size = 10

        self.reset()

    def reset(self):
        self.my_particles = []
        self.agent = None
        
        for n in range(self.number_of_particles):
            size = random.randint(20, 50)
            x = random.randint(size, width-size)
            y = random.randint(size, height-size)

            # particle = Particle((x, y), size, self.ball_density*size**2)
            particle = Particle((x, y), size, 1e-6)
            particle.colour = (100, 100, 255)
            particle.speed = 0
            particle.angle = 0

            self.my_particles.append(particle)

            self.clock = pygame.time.Clock()
            self.selected_particle = None
            self.running = True
        
        agent_x = random.randint(self.agent_size, width-self.agent_size)
        agent_y = random.randint(self.agent_size, height-self.agent_size)

        # self.agent = Particle((agent_x, agent_y), agent_size, self.agent_density*agent_size**2, name="Agent")
        self.agent = Particle((agent_x, agent_y), self.agent_size, 1, name="Agent")
        self.agent.colour = (255, 0, 0)
        self.agent.speed = 0
        self.agent.angle = 0

        self.draw_elements_on_canvas()

        return self.screen
    
    def render(self):
        self.draw_elements_on_canvas()
        return self.screen
    
    def close(self):
        pygame.quit()
        quit()

    def draw_elements_on_canvas(self):
        self.screen.fill(background_colour)
        # pygame.draw.circle(self.screen, (0,0,0), (200, 200), 200, 1)

        for particle in self.my_particles:
            particle.display(self.screen)

        pygame.draw.circle(self.screen, (0,255,0), (self.goal[0], self.goal[1]), 3, 0)

        self.agent.display(self.screen)
        
        pygame.display.flip()


    def step(self, action):
        action = action * self.v

        dx = action[0]
        dy = action[1]

        self.agent.angle = 0.5*math.pi + math.atan2(dy, dx)
        self.agent.speed = math.hypot(dx, dy) * 0.1

        self.agent.move()
        self.agent.bounce()

        self.agent.angle = 0.5*math.pi + math.atan2(dy, dx)
        self.agent.speed = math.hypot(dx, dy) * 0.1

        # print("Printing collisions with each particle")
        for particle in self.my_particles:
            collide(self.agent, particle)

        for i, particle in enumerate(self.my_particles):
            particle.move()
            particle.bounce()
            for particle2 in self.my_particles[i+1:]:
                collide(particle, particle2)

        # print("Done printing collisions with each particle")

        self.clock.tick(200)

        return self.screen, 0, False, False, {}