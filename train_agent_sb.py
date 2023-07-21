from particle_push.particle_push_simple_env import particlePushSimple
from particle_push.particle_move_env import particleMove

import numpy as np

import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps":  10_000_000,
    "env_name": "particlePushSimple",
    "Algorithm": "PPO"
}
run = wandb.init(
    project="ParticlePush-v2",
    name='PPO-500K-16AS-FixedBallPos',
    notes="10M iterations / 16 AS size / All varied.",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


def make_env():
    env = particlePushSimple()
    env = Monitor(env)  # record stats such as returns
    return env


env = DummyVecEnv([make_env])
# env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", batch_size=64, )
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=1_000,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

run.finish()


five_runs = np.zeros(shape=(400,400,3,1))

vec_env = model.get_env()
obs = vec_env.reset()
num_done = 0
for i in range(5):
    curr_run = np.zeros(shape=(5001,400,400,3))
    length = 0
    while True:
        print(length) if length % 100 == 0 else None
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        im = vec_env.render("rgb_array")
        # Append to five_runs
        curr_run[length] = im
        length += 1
        # VecEnv resets automatically
        if done:
            obs = vec_env.reset()
            print(f"Finished {num_done}")
            num_done += 1
            break
    # Convert to uint8
    curr_run = curr_run.astype(np.uint8)

    # Save only length frames
    curr_run = curr_run[:length]

    # Save the five runs as a gif
    import imageio
    imageio.mimsave(f"runs/{run.id}/trial_{i}.gif", curr_run, duration=10)
