# Multi-task Hierarchical Adversarial Inverse Reinforcement Learning

## How to config the environments:
- on Ubuntu 18.04
- python 3.8
- D4RL 1.1
- dm-control 1.0.8
- pytorch 1.7.1
- tensorboard 2.5.0
- mujoco_py 2.1.2.14
- gym 0.19.0
- seaborn
- ...


## Comparisons with Meta Imitation Learning baselines
- The algorithms are evaluated on four tasks: HalfCheetah, Walker, Ant, and Kitchen.
- To reproduce the comparison result on task 'XXX', you need to first enter the corresponding folder 'MetaHIL_XXX', where 'XXX' can be any of the four listed above.
- To run the experiment with specific algorithms:
```bash
% MH-AIRL:
python run_main.py --env_name XXX --n_traj 1000 --algo meta_hier_airl --seed YYY
% AML-IL:
python run_mamlIL_baselines.py --env_name XXX --n_traj 1000 --algo MAML_IL --seed YYY
% EMIRL:
python run_metaIRL_baselines.py --env_name XXX --n_traj 1000 --algo PEMIRL --seed YYY
% MILE
python run_metaIRL_baselines.py --env_name XXX --n_traj 1000 --algo SMILE --seed YYY
```
- 'XXX' can be one of {'HalfCheetahVel-v0', 'WalkerRandParams-v0', 'AntCell-v1', 'KitchenMetaEnv-v0'}, while the random seed 'YYY' can be 0, 1, 2, 3, or 4.

## Ablation study results
- The algorithms are evaluated on four tasks: HalfCheetah, Walker, Ant, and Kitchen.
- To reproduce the comparison result on task 'XXX', you need to first enter the corresponding folder 'MetaHIL_Ablation_XXX', where 'XXX' can be any of the four listed above.
- To run the experiment with specific algorithms:
```bash
# MH-GAIL
python run_other_ablations.py --env_name XXX --n_traj 1000 --algo meta_hier_gail --seed YYY
# Option-GAIL
python run_hierIL_baselines.py --env_name XXX --n_traj 1000 --algo option_gail --seed YYY
# DI-GAIL
python run_hierIL_baselines.py --env_name XXX --n_traj 1000 --algo DI_gail --seed YYY
# H-AIRL
python run_hierIL_baselines.py --env_name XXX --n_traj 1000 --algo hier_airl --seed YYY
```
- 'XXX' can be one of {'HalfCheetahVel-v0', 'WalkerRandParams-v0', 'AntCell-v1', 'KitchenMetaEnv-v0'}, while the random seed 'YYY' can be 0, 1, 2, 3, or 4.


## Results on the PointCell/Maze/Room
- To reproduce the visualization of the learned options on PointCell, you need to first enter the folder 'MetaHIL_Visual_Point', and then run:
```bash
# MH-AIRL
python run_main.py --algo meta_hier_airl
```
- The visualization results on PointCell will be generated at 'MetaHIL_Visual_Point/visual_result'.
- To reproduce the transfer learning results on PointMaze/Room, you need to first enter the folder 'MetaHIL_Transfer_PointMaze' or 'MetaHIL_Transfer_PointRoom'.
- To get the results with certain algorithms:
```bash
# HPPO-init
python run_main.py --env_name XXX
# HPPO
python run_ppo.py --algo 'hier_ppo' --env_name XXX
# DAC
python run_ppo.py --algo 'dac' --env_name XXX
# PPO
python run_ppo.py --algo 'ppo' --env_name XXX
```
- 'XXX' can be "PointMaze-v[0~3]" or "PointRoom-v[0~3]", where 0~3 represent four different goal locations.
- Note that the learning on PointMaze/Room is based on the pretrained model from PointCell, which is stored at the subfolder 'option_model' of 'MetaHIL_Transfer_PointMaze' and 'MetaHIL_Transfer_PointRoom'. You can add the pretrained model you get from your own training to the 'option_model' folder and test it. You only need to update the 'config.policy_model' and 'config.critic_model' in 'run_main.py' of 'MetaHIL_Transfer_PointMaze' and 'MetaHIL_Transfer_PointRoom' accordingly.

## Visualization results on HalfCheetahVel-v0
- To reproduce the visualization results on HalfCheetahVel-v0, you need to first enter the folder 'MetaHIL_Visual_HalfCheetah'. 
- To get the visualization result with offline data, you can directly run:
```bash
python visualize_dst.py
```
- You can also train the model by yourself:
```bash
# MH-AIRL
python run_main.py --algo meta_hier_airl
```

## Comparison results on WalkerRandParams-v0
- To reproduce the comparison results on WalkerRandParams-v0, you need to first enter the folder 'MetaHIL_Visual_Walker'. 
- To reproduce the plots of 'MH-AIRL', you can directly run:
```bash
# MH-AIRL
python run_main.py --algo meta_hier_airl --seed YYY
```
- 'YYY' can be 0, 1, 2, 3, or 4.
- To reproduce the plots of 'MH-AIRL-no-cnt', you need first manually change the property terms in 'default_config.py': "use_hppo" -> True, "option_with_context" -> False. Then, run:
```bash
# MH-AIRL-no-cnt
python run_main.py --algo meta_hier_airl --seed YYY
```
