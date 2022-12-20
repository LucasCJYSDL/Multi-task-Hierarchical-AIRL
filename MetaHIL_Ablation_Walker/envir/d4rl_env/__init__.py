from gym.envs.registration import register

register(
    id='mykitchen_relax-v1',
    entry_point='d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0:KitchenTaskRelaxV1',
    max_episode_steps=280,
)

register(
    id='demo_env-v0',
    entry_point='envir.d4rl_env.my_kitchen_envs:DemoParseEnv',
    max_episode_steps=500,
)

register(
    id='KitchenMetaEnv-v0',
    entry_point='envir.d4rl_env.my_kitchen_envs:KitchenMultiTask',
    max_episode_steps=500)

register(
    id='KitchenMetaEnv-v1',
    entry_point='envir.d4rl_env.my_kitchen_envs_cont:KitchenMultiTask',
    max_episode_steps=500)

register(
    id='KitchenSeqEnv-v0',
    entry_point='envir.d4rl_env.my_kitchen_envs:KitchenBottomBurner',
    max_episode_steps=250)

register(
    id='KitchenSeqEnv-v1',
    entry_point='envir.d4rl_env.my_kitchen_envs:KitchenTopBurner',
    max_episode_steps=250)

register(
    id='KitchenSeqEnv-v2',
    entry_point='envir.d4rl_env.my_kitchen_envs:KitchenLightSwitch',
    max_episode_steps=250)

register(
    id='KitchenSeqEnv-v3',
    entry_point='envir.d4rl_env.my_kitchen_envs:KitchenSlideCabinet',
    max_episode_steps=250)

register(
    id='KitchenSeqEnv-v4',
    entry_point='envir.d4rl_env.my_kitchen_envs:KitchenHingeCabinet',
    max_episode_steps=250)

register(
    id='KitchenSeqEnv-v5',
    entry_point='envir.d4rl_env.my_kitchen_envs:KitchenMicrowave',
    max_episode_steps=250)

register(
    id='KitchenSeqEnv-v6',
    entry_point='envir.d4rl_env.my_kitchen_envs:KitchenKettle',
    max_episode_steps=250)

