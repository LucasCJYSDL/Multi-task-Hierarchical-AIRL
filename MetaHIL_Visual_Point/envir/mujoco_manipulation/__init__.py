import gym

# half_cheetah_vel
gym.envs.register(
    id="HalfCheetahVel-v0",
    entry_point="envir.mujoco_manipulation.half_cheetah_vel:HalfCheetahVelEnv",
    max_episode_steps=500 # important parameter
)

gym.envs.register(
    id="WalkerRandParams-v0",
    entry_point="envir.mujoco_manipulation.walker_rand_params:WalkerRandParamsEnv",
    max_episode_steps=500 # important parameter
)

__version__ = "0.2.0"
