"""
Separate training script using MAGNNET model / task-augmented obs.
Keeps your main train.py untouched.
"""
import os
import random
import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from scripts.rllib_model_magnet import GNNPerAgentModelMAGNET
from scripts.airsim_env import rllib_env  # reuse your existing factory


def _env_creator(env_config):
    # Reuse your raw env. For MAGNNET, make sure the env provides task_pos/task_mask.
    # If you don't have native tasks yet, you can patch your env to call
    # scripts.airsim_env_magnet.add_tasks_to_obs before returning obs.
    return ParallelPettingZooEnv(rllib_env(**env_config))


def main():
    with open('config.yml', 'r') as f:
        base = yaml.safe_load(f)

    image_shape = (84, 84, 1) if base.get("train_mode", "single_rgb") == "depth" else (84, 84, 3)
    num_drones = int(base.get("num_drones", 5))
    ip_address = os.environ.get("AIRSIM_IP", "127.0.0.1")
    num_rollout_workers = int(os.environ.get("NUM_WORKERS", "1"))
    global_seed = int(os.environ.get("TRAIN_SEED", "42"))
    random.seed(global_seed)

    # Register model
    ModelCatalog.register_custom_model("gnn_per_agent_model_magnet", GNNPerAgentModelMAGNET)

    register_env("airsim_pz_magnet", _env_creator)

    algo_config = (
        PPOConfig()
        .environment(env="airsim_pz_magnet", env_config={
            "ip_address": ip_address,
            "image_shape": image_shape,
            "input_mode": base.get("train_mode", "single_rgb"),
            "num_drones": num_drones,
            "include_depth_in_cam": True,
            # Ensure your env fills task_pos/task_mask in observations
        }, disable_env_checking=True)
        .framework("torch")
        .rollouts(num_rollout_workers=num_rollout_workers)
        .resources(num_gpus=1)
        .training(
            gamma=0.99,
            lr=2e-4,
            train_batch_size=4096,
            sgd_minibatch_size=1024,
            num_sgd_iter=8,
            vf_clip_param=100.0,
            clip_param=0.2,
            entropy_coeff=0.005,
        )
        .model(custom_model="gnn_per_agent_model_magnet",
               custom_model_config={
                   "cnn_out_dim": 256,
                   "attn_dim": 64,
                   "gat_hidden": 128,
                   "gat_heads": 2,
                   "gat_layers": 1,
                   # MAGNNET params
                   "magnet_task_feat_dim": 3,
               })
    )

    algo = algo_config.debugging(seed=global_seed).build()

    # Simple training loop placeholder
    for i in range(20):
        res = algo.train()
        print(f"Iter {i+1}", {k: res[k] for k in ("episode_reward_mean", "episode_len_mean") if k in res})


if __name__ == "__main__":
    main()

