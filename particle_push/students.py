import numpy as np
    
# Requires access to the environment parameters and the environment to function
def whiteBoxStudent(state, env_params, env):
    # Says whether each ball is in its goal
    ball_status = env.game.get_info()
    # Find the smallest index of a ball that isn't in its goal
    goal_ball_idx = np.argmin(ball_status)
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