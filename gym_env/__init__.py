from gym.envs.registration import register


########################################################################################
# GRIDWORLD
register(
    id='IPD-v0',
    entry_point='gym_env.ipd.ipd_env:IPDEnv',
    kwargs={'args': None},
    max_episode_steps=150
)
