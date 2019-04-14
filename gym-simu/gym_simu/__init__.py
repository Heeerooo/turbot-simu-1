from gym.envs.registration import register

register(
    id='simu-v0',
    entry_point='gym_simu.envs:SimuEnv',
)