from particle_push.particle_push_env import particlePush
import numpy as np
# from particle_push.students import whiteBoxStudent
# import json

# from ray import tune

experiment_path = './results/Training_final/' #SAC_particlePush_e1e23_00000_0_2023-07-07_12-17-15/'
# checkpoint_path = './results/Training_final/SAC_particlePush_e1e23_00000_0_2023-07-07_12-17-15/checkpoint_001100/'
checkpoint_path = './results/Training_wmw/SAC_particlePush_fd875_00000_0_2023-07-07_15-24-08/checkpoint_022100/'

from ray.rllib.algorithms.algorithm import Algorithm

# Use the Algorithm's `from_checkpoint` utility to get a new algo instance
# that has the exact same state as the old one, from which the checkpoint was
# created in the first place:
my_new_ppo = Algorithm.from_checkpoint(checkpoint_path)

from ray.rllib.policy.policy import Policy

my_policy = Policy.from_checkpoint(checkpoint_path)['default_policy']

# obs = np.array([[0.,0.],[0.,0.]], dtype=np.float32)

# action = my_policy.compute_single_action(obs)
# print(action)

for _ in range(20):
    env = particlePush(render_mode = 'human', num_balls = 1)

    obs = env.reset()

    while True:
        # action = whiteBoxStudent(obs, env.get_params(), env)
        # action = env.action_space.sample()
        action, _, _ = my_policy.compute_single_action(obs)

        obs, reward, done, info = env.step(action)
        print(reward)
        env.render()

        if done:
            break