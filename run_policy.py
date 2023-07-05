import yaml
from particle_push.particle_teacher_env import particlePushTeacher
from particle_push.particle_push_env import particlePush
from particle_push.students import *
from particle_push.teachers import *


def run_teacher_student_game(protagonist, teacher=randomTeacher, antagonist=None, visualize=False):
    if antagonist is None:
        env = particlePushTeacher(protagonist, render_mode = 'rgb_array')
        obs = env.reset()

        while True:
            action = teacher(obs)
            obs, reward, term, trunc, info = env.step(action)

            if visualize:
                env.render()

            if term or trunc:
                break

        rewards = {'protagonist_reward':env.get_agent_reward(), 'teacher_reward':env.get_teacher_reward()}
        env.close()
        return rewards
    else:
        env_antagonist = particlePushTeacher(antagonist, render_mode = 'rgb_array')
        obs_ant = env_antagonist.reset()
        env_protagonist = particlePushTeacher(protagonist, render_mode = 'rgb_array')
        obs_pro = env_protagonist.reset()

        action_ant = teacher(obs_ant)
        obs_ant, reward_ant, term_ant, trunc_ant, info_ant = env_antagonist.step(action_ant)

        final_pos = env_antagonist.game.get_state()[1]
        action_pro = action_ant
        action_pro['ball_goals'] = final_pos

        obs_pro, reward_pro, term_pro, trunc_pro, info_pro = env_protagonist.step(action_pro)

        rewards = {'protagonist_reward':env_protagonist.get_agent_reward(), 'antagonist_reward':env_antagonist.get_teacher_reward()}
        env_protagonist.close()
        env_antagonist.close()
        return rewards
