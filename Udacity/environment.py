from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

def giveEnv(location):
    unity_env = UnityEnvironment(location)
    env = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=True)  # , allow_multiple_obs=True #
    return env
