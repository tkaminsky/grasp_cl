import numpy as np
import random

# Randomly samples a valid environment 
def randomTeacher():
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