from particle_push.particle_push_env import particlePush
from particle_push.students import whiteBoxStudent

from ray import tune

tune.run(
    "SAC", # reinforced learning agent
    name = "Training_wmw",
    # to resume training from a checkpoint, set the path accordingly:
    # resume = True, # you can resume from checkpoint
    # restore = r'.\ray_results\Example\SAC_RocketMeister10_ea992_00000_0_2020-11-11_22-07-33\checkpoint_3000\checkpoint-3000',
    checkpoint_freq = 100,
    checkpoint_at_end = True,
    local_dir = r'./results/',
    config={
        "env": particlePush,
        "num_workers": 30,
        "num_cpus_per_worker": 0.5,
        "env_config": {
            "num_balls" : None,
            "ball_sizes" : None,
            "ball_inits" : None,
            "agent_init" : None,
            "ball_goals" : None,
            "render_mode" : 'human'
        }
        },
    stop = {
        "timesteps_total": 5_000_000,
        },
    )

# for _ in range(20):
#     env = particlePush(render_mode = 'human', num_balls = 1)

#     obs = env.reset()

#     while True:
#         # action = whiteBoxStudent(obs, env.get_params(), env)
#         action = env.action_space.sample()

#         obs, reward, term, trunc, info = env.step(action)
#         print(reward)
#         env.render()

#         if term or trunc:
#             break