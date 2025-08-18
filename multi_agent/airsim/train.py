import sys
import os

# Add modified_libs to Python path to use modified libraries
# modified_libs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'modified_libs')
# sys.path.insert(0, modified_libs_path)

import time
import yaml
import numpy as np
import random

# ===== OLD SB3 APPROACH (commented, kept for rollback) =====
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
# from scripts.network import CustomCombinedExtractor ,MultiAgentGNNPolicy ,GNNCombinedExtractor
# from stable_baselines3.common.env_checker import check_env
# from pettingzoo.test import parallel_api_test
# from scripts.airsim_env import AirSimDroneEnv, petting_zoo
#
# with open('config.yml', 'r') as f:
#     config = yaml.safe_load(f)
# image_shape = (84,84,1) if config["train_mode"]=="depth" else (84,84,3)
# env = petting_zoo(
#         ip_address="127.0.0.1",
#         image_shape=image_shape,
#         input_mode=config["train_mode"],
#         num_drones=config["num_drones"]
#         )
# policy_kwargs = dict(
#     features_extractor_class=GNNCombinedExtractor,
#     num_drones=config["num_drones"]
# )
# model = PPO(
#     MultiAgentGNNPolicy,
#     env,
#     learning_rate=0.0001,
#     batch_size=256,
#     clip_range=0.2,
#     verbose=1,
#     seed=42,
#     device="cuda",
#     tensorboard_log="./tb_logs/",
#     policy_kwargs=policy_kwargs,
# )
# ==========================================================

# ===== RLlib PPO (IPPO/MAPPO) with PettingZoo =====
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune.logger import TBXLoggerCallback

# Use the raw PettingZoo env factory (no Supersuit vec wrappers)
from scripts.airsim_env import rllib_env
from scripts.rllib_model import GNNPerAgentModel

# Register custom model
ModelCatalog.register_custom_model("gnn_per_agent_model", GNNPerAgentModel)

# Load configs
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

image_shape = (84,84,1) if config.get("train_mode", "single_rgb")=="depth" else (84,84,3)
num_drones = int(config.get("num_drones", 5))

# Allow overrides via environment for cluster use
ip_address = os.environ.get("AIRSIM_IP", "127.0.0.1")
num_rollout_workers = int(os.environ.get("NUM_WORKERS", "1"))
global_seed = int(os.environ.get("TRAIN_SEED", "42"))

# Set seeds for reproducibility (to the extent RLlib supports it)
np.random.seed(global_seed)
random.seed(global_seed)

# Register RLlib env using ParallelPettingZooEnv (native parallel wrapper)

def _env_creator(env_config):
    parallel_env = rllib_env(**env_config)
    return ParallelPettingZooEnv(parallel_env)

register_env("airsim_pz", _env_creator)

# Build a temporary env to obtain per-agent spaces
# Define exact spawn points (meters) once and reuse for both temp and training envs
spawn_points = {
    # UE4 cm -> meters
    "drone0": (-72.97, 10.0, -1.2),
    "drone1": (-72.97, 20.0, -1.2),
    "drone2": (-72.97, 15.0, -1.2),
    "drone3": (-72.97, 28.0, -1.2),
}

_tmp = rllib_env(
    ip_address=ip_address,
    image_shape=image_shape,
    input_mode=config.get("train_mode", "single_rgb"),
    num_drones=num_drones,
    include_depth_in_cam=True,
    spawn_points=spawn_points,
)
obs_space = _tmp.observation_spaces["drone0"]
act_space = _tmp.action_spaces["drone0"]
del _tmp

algo_config = (
    PPOConfig()
    .environment(env="airsim_pz", env_config={
        "ip_address": ip_address,
        "image_shape": image_shape,
        "input_mode": config.get("train_mode", "single_rgb"),
        "num_drones": num_drones,
        "include_depth_in_cam": True,
        "spawn_points": spawn_points,
    }, disable_env_checking=True)
    .framework("torch")
    .rollouts(num_rollout_workers=num_rollout_workers)
    .debugging(seed=global_seed, logger_config={"type": "ray.tune.logger.TBXLogger", "logdir": "tb_logs/rllib"})
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
    .multi_agent(
        policies={
            "shared": (
                None,
                obs_space,
                act_space,
                {
                    "model": {
                        "custom_model": "gnn_per_agent_model",
                        "custom_model_config": {
                            "cnn_out_dim": 256,
                            "attn_dim": 64,
                            "action_dim": 4,
                        },
                    }
                },
            )
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared",
    )
    .resources(num_gpus=1 if __import__('torch').cuda.is_available() else 0)
)

algo = algo_config.build()

import glob, os
# ckpts = sorted(glob.glob("saved_policy/**/checkpoint_*", recursive=True), key=os.path.getmtime)
# algo.restore('saved_policy\\rllib_checkpoint.json')

# If a specific restore file exists or RESUME_FROM is provided, restore; otherwise train from scratch
restore_env_file = os.environ.get("RESUME_FROM", "")
if restore_env_file:
    restore_file = restore_env_file
else:
    restore_file = os.path.join("saved_policy//final_1755189646", "rllib_checkpoint.json")

if os.path.isfile(restore_file):
    restore_dir = os.path.dirname(restore_file)
    print(f"Restoring from: {restore_dir}")
    algo.restore(restore_dir)
else:
    print("Training from scratch (no restore file found)")
# Simple training loop with periodic and final checkpointing
try:
    for i in range(70):
        print(f"--- Iteration {i+1}/70: collecting rollouts and updating PPO ---")
        result = algo.train()
        print(f"+++ PPO policy updated at iteration {i+1} +++")
        print(pretty_print(result))
        # Save every 5 iterations (5, 10, 15, ...), into unique subfolders
        if (i + 1) % 5 == 0:
            save_dir = os.path.join("saved_policy", f"iter_{i+1:04d}")
            os.makedirs(save_dir, exist_ok=True)
            chkpt = algo.save(checkpoint_dir=save_dir)
            print(f"Checkpoint saved (iter {i+1}) at: {chkpt}")
except KeyboardInterrupt:
    # Save on manual stop
    int_dir = os.path.join("saved_policy", f"interrupt_{int(time.time())}")
    os.makedirs(int_dir, exist_ok=True)
    chkpt = algo.save(checkpoint_dir=int_dir)
    print(f"Checkpoint saved on interrupt at: {chkpt}")
finally:
    # Always save a final checkpoint when the loop ends
    final_dir = os.path.join("saved_policy", f"final_{int(time.time())}")
    os.makedirs(final_dir, exist_ok=True)
    chkpt = algo.save(checkpoint_dir=final_dir)
    print(f"Final checkpoint saved at: {chkpt}")
