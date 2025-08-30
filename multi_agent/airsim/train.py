import sys
import os
import time
import yaml
import numpy as np
import random

# ===== RLlib PPO with PettingZoo =====
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

# Import your custom environment and model
from scripts.airsim_env import rllib_env
from scripts.rllib_model import GNNPerAgentModel

# Register custom model
ModelCatalog.register_custom_model("gnn_per_agent_model", GNNPerAgentModel)

print("=== Starting Fresh Training Setup ===")

# Load configs
print("1. Loading configuration...")
try:
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded: {config}")
except Exception as e:
    print(f"✗ Failed to load config: {e}")
    sys.exit(1)

# Set basic parameters
image_shape = (84,84,1) if config.get("train_mode", "single_rgb")=="depth" else (84,84,3)
num_drones = int(config.get("num_drones", 4))
ip_address = os.environ.get("AIRSIM_IP", "127.0.0.1")
num_rollout_workers = int(os.environ.get("NUM_WORKERS", "0"))
global_seed = int(os.environ.get("TRAIN_SEED", "42"))

print(f"✓ Parameters set:")
print(f"  - Image shape: {image_shape}")
print(f"  - Number of drones: {num_drones}")
print(f"  - IP address: {ip_address}")
print(f"  - Rollout workers: {num_rollout_workers}")
print(f"  - Seed: {global_seed}")

# Set seeds for reproducibility
np.random.seed(global_seed)
random.seed(global_seed)

spawn_points = None  # keep drones where the map starts when pressing Play

print("2. Preparing environment and algorithm...")
obs_space = None
act_space = None
try:
    tmp_env = rllib_env(
        ip_address=ip_address,
        image_shape=image_shape,
        input_mode=config.get("train_mode", "single_rgb"),
        num_drones=num_drones,
        include_depth_in_cam=True,
        spawn_points=spawn_points,
    )
    obs_space = tmp_env.observation_spaces["drone0"]
    act_space = tmp_env.action_spaces["drone0"]
    del tmp_env
except Exception as e:
    print(f"✗ Minimal env probe failed: {e}")
    sys.exit(1)

print("5. Setting up RLlib environment...")
try:
    # Register environment
    def _env_creator(env_config):
        parallel_env = rllib_env(**env_config)
        return ParallelPettingZooEnv(parallel_env)
    
    register_env("airsim_pz", _env_creator)
    print("✓ Environment registered successfully")
    
except Exception as e:
    print(f"✗ Failed to register environment: {e}")
    sys.exit(1)

print("6. Creating algorithm configuration...")
try:
    algo_config = (
        PPOConfig()
        .environment(env="airsim_pz", env_config={
            "ip_address": ip_address,
            "image_shape": image_shape,
            "input_mode": config.get("train_mode", "single_rgb"),
            "num_drones": num_drones,
            "include_depth_in_cam": True,
            "spawn_points": spawn_points,
        })
        .framework("torch")
        .env_runners(
            num_env_runners=num_rollout_workers,
            sample_timeout_s=600,
            rollout_fragment_length=8,       # much smaller fragments to flush samples more frequently
        )
        .rollouts(
            batch_mode="truncate_episodes"   # allow partial episodes to be emitted
        )
        .reporting(
            min_sample_timesteps_per_iteration=32
        )
        .debugging(seed=global_seed, logger_config={"type": "ray.tune.logger.TBXLogger", "logdir": "tb_logs/rllib"})
        .training(
            gamma=0.99,
            lr=2e-4,
            train_batch_size=512,  # Reduced from 1024 for less GPU usage
            sgd_minibatch_size=128,  # Reduced from 256 for less GPU usage
            num_sgd_iter=2,  # Reduced from 4 for less GPU usage
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
        .resources(
            num_gpus=1 if __import__('torch').cuda.is_available() else 0,  # Use only half GPU
        )
    )
    
    print("✓ Algorithm configuration created successfully")
    print("  - Reduced batch sizes for lower memory usage")
    
except Exception as e:
    print(f"✗ Failed to create algorithm config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("7. Building algorithm...")
try:
    algo = algo_config.build()
    print("✓ Algorithm built successfully")
    
except Exception as e:
    print(f"✗ Failed to build algorithm: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("7.5. Checking for existing checkpoints...")
checkpoint_dir = None
start_iteration = 0

# Look for the most recent checkpoint
import glob
checkpoints = glob.glob("saved_policy/iter_0125")
if checkpoints:
    # Sort by iteration number and get the latest
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1]))[-1]
    checkpoint_dir = latest_checkpoint
    start_iteration = int(latest_checkpoint.split('_')[-1])
    print(f"✓ Found checkpoint: {checkpoint_dir}")
    print(f"  - Last completed iteration: {start_iteration}")
    
    
    algo.restore(checkpoint_dir)
    print("✓ Restored successfully!")
    start_iteration += 1  # Start from next iteration
  




# Simple training loop
try:
    # Track global iteration count to avoid checkpoint overwriting
    global_iteration = 0
    if checkpoint_dir and "iter_" in checkpoint_dir:
        try:
            global_iteration = int(checkpoint_dir.split("iter_")[1])
            print(f"Continuing from global iteration: {global_iteration}")
        except:
            global_iteration = 0
    
    for i in range(start_iteration, 80):  # full planned iterations
        current_global_iteration = global_iteration + i + 1
        print(f"\n--- Iteration {current_global_iteration}/80 ---")
        
        try:
            result = algo.train()
            print(f"✓ Training iteration {current_global_iteration} completed")
            
            # Show key metrics
            episodes = result.get('env_runners', {}).get('episodes_this_iter', 0)
            timesteps = result.get('num_env_steps_sampled_this_iter', 0)
            healthy_workers = result.get('num_healthy_workers', 0)
            
            print(f"  - Episodes: {episodes}")
            print(f"  - Timesteps: {timesteps}")
            print(f"  - Healthy workers: {healthy_workers}")
            
            if timesteps == 0:
                print("  ⚠️  WARNING: No data collected this iteration!")

            # Every 2 iterations: print reward/learning snapshot
            if (i + 1) % 2 == 0:
                er_mean = result.get('env_runners', {}).get('episode_reward_mean')
                el_mean = result.get('env_runners', {}).get('episode_len_mean')
                hist = result.get('env_runners', {}).get('hist_stats', {})
                hist_rewards = hist.get('episode_reward', []) or []
                hist_lengths = hist.get('episode_lengths', []) or []
                r_min = result.get('env_runners', {}).get('episode_reward_min')
                r_max = result.get('env_runners', {}).get('episode_reward_max')
                total_env_steps = result.get('num_env_steps_sampled_lifetime', result.get('num_env_steps_sampled', 0))
                total_agent_steps = result.get('agent_timesteps_total', result.get('num_agent_steps_sampled', 0))
                pr_mean = (result.get('env_runners', {}).get('policy_reward_mean', {}) or {}).get('shared', None)

                print("\n=== Checkpoint summary (every 2 iters) ===")
                print(f"  Iteration: {current_global_iteration}")
                print(f"  Episode reward mean: {er_mean}")
                print(f"  Episode length mean: {el_mean}")
                print(f"  Reward range: min={r_min}, max={r_max}")
                if pr_mean is not None:
                    print(f"  Policy 'shared' reward mean: {pr_mean}")
                print(f"  Episodes recorded (hist): {len(hist_rewards)}; Steps recorded (hist): {len(hist_lengths)}")
                print(f"  Total env steps: {total_env_steps}; Total agent steps: {total_agent_steps}")
                print("========================================\n")
            
            # Save checkpoint every 5 iterations
            if current_global_iteration % 5 == 0:
                save_dir = os.path.join("saved_policy", f"iter_{current_global_iteration:04d}")
                os.makedirs(save_dir, exist_ok=True)
                chkpt = algo.save(checkpoint_dir=save_dir)
                print(f"  - Checkpoint saved: {chkpt}")
                
        except Exception as e:
            print(f"✗ Training iteration {current_global_iteration} failed: {e}")
            print("Continuing to next iteration...")
            continue
            
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Save final checkpoint
    print("\n9. Saving final checkpoint...")
    try:
        final_dir = os.path.join("saved_policy", f"final_{int(time.time())}")
        os.makedirs(final_dir, exist_ok=True)
        chkpt = algo.save(checkpoint_dir=final_dir)
        print(f"✓ Final checkpoint saved: {chkpt}")
    except Exception as e:
        print(f"✗ Failed to save final checkpoint: {e}")

print("\n=== Training Complete ===")
