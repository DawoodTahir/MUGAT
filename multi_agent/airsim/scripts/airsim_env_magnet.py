import os


import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import gymnasium
import supersuit as ss
from itertools import combinations
from . import airsim
from pettingzoo import ParallelEnv
from typing import Optional
from gymnasium.utils import EzPickle, seeding

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def env(**kwargs):
    """SB3/SuperSuit vectorized env (kept for backward compat)."""
    env = AirSimDroneEnv(**kwargs)
    env = ss.black_death_v3(env)
    #env = ss.frame_stack_v2(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    return env

# For existing SB3 code
petting_zoo = env

# New: Raw PettingZoo env factory for RLlib (no vector wrappers)
def rllib_env(**kwargs) -> ParallelEnv:
    return AirSimDroneEnv(**kwargs)


class AirSimDroneEnv(ParallelEnv, EzPickle):
    metadata = {'name': 'drones', 'render_modes': ['human']}

    def __init__(self,  
                 ip_address, 
                 image_shape, 
                 input_mode, 
                 num_drones,
                 include_depth_in_cam: bool = True,
                 agent_offsets: Optional[dict] = None,
                 spawn_points: Optional[dict] = None,
                 comm_mode: Optional[str] = None,
                 allocation_mode: Optional[str] = None,
                 task_names: Optional[list] = None,
                 task_value_map: Optional[dict] = None,
                 use_noisy_pos: Optional[bool] = True,
                 gps_noise_std: Optional[list] = None,
                 enable_peer_sidestep: Optional[bool] = False,
                 global_time_budget_steps: Optional[int] = None,
                 stagnation_timeout: Optional[int] = None,
                 time_budget_warmup_episodes: Optional[int] = None,
                 stagnation_warmup_episodes: Optional[int] = None,
                 disable_lidar: Optional[bool] = None,
                 ):
        EzPickle.__init__(self,  
                 ip_address = ip_address, 
                 image_shape = image_shape, 
                 input_mode = input_mode, 
                 num_drones = num_drones
                 )
        
        # Settings
        self.image_shape = image_shape
        self.input_mode = input_mode
        self.num = num_drones
        # Feature toggles (safe defaults)
        self.disable_lidar = False

        # Init
        self.drone = airsim.MultirotorClient(ip=ip_address)
        # Comms mode: full or none (default full; can be passed in constructor)
        self.comm_mode = (comm_mode or "full").lower().strip()
        # Allocation mode: 'heuristic' (ETA-based) or 'learned' (policy bids)
        self.allocation_mode = os.environ.get("ALLOCATION_MODE", "heuristic").lower().strip()

        # Task registry (fixed names; can be overridden via constructor)
        self.task_names = task_names if task_names is not None else [
            "Cylinder5", "Cylinder2_5", "Cylinder6", "Cylinder3", "Cylinder4"
        ]
        self.num_tasks = max(1, len(self.task_names))
        # Per-task value hashmap keyed by task name; default uniform 1.0
        self.task_value_map = {}
        if isinstance(task_value_map, dict) and len(task_value_map) > 0:
            for nm in self.task_names:
                try:
                    self.task_value_map[nm] = float(task_value_map.get(nm, 1.0))
                except Exception:
                    self.task_value_map[nm] = 1.0
        else:
            # Default randomized values {20,40,60,80,100}; first task gets 100
            try:
                values_pool = [80.0, 80.0, 80.0, 80.0, 80.0]
                # Deterministic shuffle for stability across runs
                rng = np.random.RandomState(12345)
                remaining_vals = [v for v in values_pool if v != 100.0]
                rng.shuffle(remaining_vals)
                # Assign values (first task highest by default)
                for idx, nm in enumerate(self.task_names):
                    if idx == 0:
                        self.task_value_map[nm] = 100.0
                    else:
                        if len(remaining_vals) == 0:
                            self.task_value_map[nm] = 20.0
                        else:
                            self.task_value_map[nm] = float(remaining_vals.pop())
            except Exception:
                for nm in self.task_names:
                    self.task_value_map[nm] = 20.0
        # Index-aligned values list for fast access
        self.task_value = [float(self.task_value_map.get(nm, 1.0)) for nm in self.task_names]
        # Success reward policy: fixed 1000 + task_value bonus
        self.success_base_reward = 1000.0

        # PettingZoo variables
        self.possible_agents = ["drone"+str(i) for i in range(0,num_drones)]
        self.agents = self.possible_agents[:]
        self.truncations = None
        # Debug flag to gate verbose prints
        self.debug = True
        self.collision_time = None
        self.reward = None
        # self.done = None  # old
        self.terminations = None  # new ParallelEnv API
        self.obj = None
        self.max_steps = 200  # reduced from 250 since drones will move faster now
        self.current_step = None
        # Global time budget and stagnation timeout
        self.global_time_budget_steps = int(global_time_budget_steps) if global_time_budget_steps is not None else 2000
        self.stagnation_timeout = int(stagnation_timeout) if stagnation_timeout is not None else 200
        self.time_budget_warmup_episodes = int(time_budget_warmup_episodes) if time_budget_warmup_episodes is not None else 0
        self.stagnation_warmup_episodes = int(stagnation_warmup_episodes) if stagnation_warmup_episodes is not None else 0
        self._last_completion_step = 0

        # Observation space - FIXED for PettingZoo compatibility
        self.observation_spaces = gymnasium.spaces.Dict(
                {
                    id:gymnasium.spaces.Dict(
                        {
                            "cam":gymnasium.spaces.Box(
                                low=0, 
                                high=255, 
                                shape=self.image_shape, 
                                dtype=np.uint8
                            ),
                            # Fixed tasks (K=self.num_tasks): positions and validity mask
                            "task_pos": gymnasium.spaces.Box(
                                low=-1e6,
                                high=1e6,
                                shape=(self.num_tasks, 3),
                                dtype=np.float32,
                            ),
                            "task_mask": gymnasium.spaces.Box(
                                low=0.0,
                                high=1.0,
                                shape=(self.num_tasks,),
                                dtype=np.float32,
                            ),
                            # New: LiDAR polar bins (32) as ranges in meters
                            "lidar": gymnasium.spaces.Box(
                                low=0.0,
                                high=1000.0,
                                shape=(32,),
                                dtype=np.float32,
                            ),
                            # Per-agent safety scalars: [ttc_pair_min, ttc_forward]
                            "ttc": gymnasium.spaces.Box(
                                low=0.0,
                                high=1e6,
                                shape=(2,),
                                dtype=np.float32,
                            ),
                            "pos":gymnasium.spaces.Box(
                                low=-500.0, 
                                high=500.0, 
                                shape=(3,), 
                                dtype=np.float32
                            ),
                            
                            # New: legacy team positions (kept for backward compat)
                            "team_pos": gymnasium.spaces.Box(
                                low=-500.0,
                                high=500.0,
                                shape=(self.num, 3),
                                dtype=np.float32,
                            ),
                            # Unified per-peer comm: [dx,dy,dz,collision, task_id_norm, bid, eta, commit]
                            "team_comm": gymnasium.spaces.Box(
                                low=-1e9,
                                high=1e9,
                                shape=(self.num, 8),
                                dtype=np.float32,
                            ),
                            # New: validity mask for comm packets (1=valid, 0=missing)
                            "team_comm_mask": gymnasium.spaces.Box(
                                low=0.0,
                                high=1.0,
                                shape=(self.num,),
                                dtype=np.float32,
                            ),
                            # New: agent index (normalized 0..1) for symmetry breaking
                            "agent_idx": gymnasium.spaces.Box(
                                low=0.0,
                                high=1.0,
                                shape=(1,),
                                dtype=np.float32,
                            ),
                            # Global remaining time (0..1) for time awareness
                            "remaining_time": gymnasium.spaces.Box(
                                low=0.0,
                                high=1.0,
                                shape=(1,),
                                dtype=np.float32,
                            ),
                        }
                    )
                    for id in self.possible_agents
                }
            )
        
        # Tasks
        # Action space (lr, fb, ud, yaw_rate, bids[0:K]) normalized to [-1, 1]
        self.action_spaces = gymnasium.spaces.Dict(
                {
                    id:gymnasium.spaces.Box(
                        low=np.concatenate([np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32), -np.ones((self.num_tasks,), dtype=np.float32)]),
                        high=np.concatenate([np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32),  np.ones((self.num_tasks,), dtype=np.float32)]),
                        shape=(4 + self.num_tasks,), 
                        dtype=np.float32
                    )
                    for id in self.possible_agents
                }
            )
        
        # Control and comm parameters - OPTIMIZED for actual movement
        self.control_dt = 1.2  # longer dt per request
        self.max_xy_speed = 2.0  # reduced to approximate DJI Tello indoor horizontal speed
        self.max_z_speed = 0.5   # slower vertical motion to keep altitude low
        self.max_yaw_rate_deg = 3.0  # further reduce yaw authority to prioritize translation
        self.smooth_alpha = 0.10  # keep smoothing light; reduce integration lag
        # Warmup scheduling for smoothing (reduce at start to encourage visible motion)
        self._episode_index = 0
        self._smooth_warmup_episodes = 2  # reduced from 3
        self._smooth_warmup_value = 0.1  # increased from 0.02
        self._last_action = {i: np.zeros(4 + self.num_tasks, dtype=np.float32) for i in self.possible_agents}
        # Remove extra velocity low-pass; rely on action smoothing only
        self._prev_cmd_v = {i: np.zeros(3, dtype=np.float32) for i in self.possible_agents}
        # Last policy bids per agent (for learned allocation)
        self._last_bids = {i: np.zeros((self.num_tasks,), dtype=np.float32) for i in self.possible_agents}
        # Cache of last forward range from LiDAR (for TTC penalty)
        self._last_forward_range = {i: 1e6 for i in self.possible_agents}
        # Cache of last pairwise TTC (min over peers) for reward penalty
        self._last_ttc_pair = {i: 1e6 for i in self.possible_agents}
        # Cache of last heading-aligned range from LiDAR (any-direction TTC)
        self._last_heading_range = {i: 1e6 for i in self.possible_agents}
        # Success dwell counters per agent
        self._success_hold = {i: 0 for i in self.possible_agents}

        # Reward shaping and safety tuning - OPTIMIZED for stable learning
        self.gamma = 0.99  # discount used for potential-based shaping
        # Removed unused shaping hyperparameters
        self.time_penalty = 0.001  # reduced from 0.007 to prevent penalty accumulation
        self.soft_wall_scale = 0.05  # reduced from 0.2 for much gentler wall penalty
        self.soft_wall_radius = 5.0  # reduced from 8.0 for smaller penalty zone
        self.near_goal_m = 35.0  # widened positive zone to engage formation shaping earlier
        self.align_k = 1.5  # stronger alignment reward near goal
        self.safe_spacing_m = 2.5  # slightly reduced to allow closer passing
        self.spacing_scale = 1.5  # softer spacing penalty
        # Pairwise TTC early-warning penalty (applies only when beyond spacing threshold)
        self.ttc_time_threshold_s = 1.5  # nudge later
        self.ttc_penalty_scale = 0.12    # softer TTC early-warning
        self.success_radius_m = 2.0  # legacy proximity success (kept for shaping)
        # Require landing within target perimeter for final success
        self.require_landing_for_success = True
        self.landing_radius_m = 0.6  # XY radius around target for successful landing
        # Require a short dwell in success state to avoid fly-by
        self.success_dwell_steps = 2  # with control_dt=1.0 → ≈2s hold
        # Single ceiling: max flyable height (NED). z >= -2.0 allowed; z < -2.0 (higher) discouraged
        self.ceiling_z = -2.0
        self.ceiling_penalty_per_meter = 0.3
        # Progress weighting and formation gate
        self.progress_w_far = 100.0  # stronger pull toward task center when far
        self.progress_w_near = 70.0  # stronger pull toward formation/micro-goal when near
        # Mild penalty when moving away from goal (progress < 0)
        self.away_penalty_far_k = 10.0
        self.away_penalty_near_k = 7.0
        # Enable formation shaping only when within this distance to task center
        # Use near_goal_m as default gate radius
        self.form_gate_enable_m = float(self.near_goal_m)
        # Optional per-agent formation offsets in world frame (x,y,z)
        # Defaults to zero offset for all agents
        self.agent_offsets = {}
        if agent_offsets is None:
            for aid in self.possible_agents:
                self.agent_offsets[aid] = np.zeros(3, dtype=np.float32)
        else:
            # Accept dict of agent->(x,y,z) or list aligned with possible_agents
            if isinstance(agent_offsets, dict):
                for aid in self.possible_agents:
                    off = agent_offsets.get(aid, (0.0, 0.0, 0.0))
                    self.agent_offsets[aid] = np.array(off, dtype=np.float32)
            else:
                # Fallback: treat as list/sequence
                for idx, aid in enumerate(self.possible_agents):
                    try:
                        off = agent_offsets[idx]
                    except Exception:
                        off = (0.0, 0.0, 0.0)
                    self.agent_offsets[aid] = np.array(off, dtype=np.float32)

        # Agent heterogeneity: per-agent speed scaling (1.0 by default)
        self.agent_speed_scale = {aid: 1.0 for aid in self.possible_agents}

        # Comm parameters
        self.comm_dim = 4  # [x,y,z,collision]
        # Reduce comm noise for faster learning
        self.comm_noise_std = 0.0
        self.comm_drop_prob = 0.0
        self.comm_delay_prob = 0.0
        self._comm_prev = {i: np.zeros((self.num, self.comm_dim), dtype=np.float32) for i in self.possible_agents}
        self._comm_prev_mask = {i: np.zeros((self.num,), dtype=np.float32) for i in self.possible_agents}
        # Peer avoidance controls
        self.enable_peer_sidestep = bool(enable_peer_sidestep)
        # LiDAR config
        self.max_lidar_points = 2048
        self._lidar_sensor_names = ["LidarSensor1", "Lidar1", "Lidar", "lidar", "LidarSensor"]
        # Effective LiDAR cap distance (meters) for normalization in observations
        self.lidar_cap_m = 12.0
        # Bid shaping
        self.bid_temp = 0.5  # <1.0 shrinks bid amplitude to avoid early saturation
        self.bid_entropy_coef = 0.005  # small reward bonus for diverse bids
        self.bid_ema_alpha = 0.8  # EMA factor for bids
        self._bid_ema = {i: np.zeros((self.num_tasks,), dtype=np.float32) for i in self.possible_agents}

        # Geofence (indoor-friendly). XY matches target boundary; Z clamp disabled for debugging
        self.x_min, self.x_max = -150.0, 150.0
        self.y_min, self.y_max = -245.0, 100.0
        # Z clamp disabled; keep placeholders
        self.z_min, self.z_max = -1e9, 1e9

        # Track geofence hits per agent for hard penalties
        self._hit_geofence = {i: False for i in self.possible_agents}

        # Sensor fusion option: include depth into cam channels (keep cam shape HxWx3)
        self.include_depth_in_cam = include_depth_in_cam
        # Exact spawn points only (dict {agent_id: (x,y,z)} or list aligned with agents)
        self.spawn_points = spawn_points
        # GPS-like noisy position toggle and std (meters)
        self.use_noisy_pos = bool(use_noisy_pos) if use_noisy_pos is not None else False
        if gps_noise_std is None:
            self.gps_noise_std = np.array([0.15, 0.15, 0.30], dtype=np.float32)
        else:
            try:
                arr = np.array(gps_noise_std, dtype=np.float32).reshape(3,)
                self.gps_noise_std = arr
            except Exception:
                self.gps_noise_std = np.array([0.15, 0.15, 0.30], dtype=np.float32)
        
        # Task/allocation state (must be set BEFORE setup_flight)
        # Allocation mode: learned only (no heuristic fallback)
        self.allocation_mode = (allocation_mode or "learned").lower().strip()
        # Default coalition size r_t per task (length = num_tasks)
        default_rt = 1
        self.task_required = [default_rt for _ in range(self.num_tasks)]
        self._tasks_pos = np.zeros((self.num_tasks, 3), dtype=np.float32)
        self._task_mask = np.zeros((self.num_tasks,), dtype=np.float32)
        # Persistent queue of under-filled ("free") tasks (FIFO)
        self._task_queue = []
        # Per-task success tracking and completion flags
        self._task_success_count = [0 for _ in range(self.num_tasks)]
        self._task_completed = [False for _ in range(self.num_tasks)]
        # Agent success accounted flag to avoid double counting
        self._agent_success_recorded = {aid: False for aid in self.possible_agents}
        self._assigned_task = {aid: -1 for aid in self.possible_agents}
        self._assigned_task_prev = {aid: -1 for aid in self.possible_agents}
        self._commit_lock_until = {aid: 0 for aid in self.possible_agents}
        self._intent_fields = {aid: np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32) for aid in self.possible_agents}
        # Track reallocation events to expose in Infos for one step
        self._realloc_mark = {aid: None for aid in self.possible_agents}
        # Per-agent record of tasks already credited for success (to allow multi-task)
        self._agent_completed_tasks = {aid: set() for aid in self.possible_agents}

        # Setup flight and set seed
        self.setup_flight()
        
        # CRITICAL: Arm all drones before accepting commands
        self._arm_all_drones()
        
        self._seed(42)
        
        # (Debug env self-test removed to speed up training; rely on logs instead)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _arm_all_drones(self):
        """Arm all drones so they can accept movement commands"""
        if self.debug:
            print("🔧 Arming all drones...")
        
        # Ensure simulator is running (not paused)
        try:
            self.drone.simPause(False)
        except Exception:
            pass

        for drone_name in self.possible_agents:
            try:
                # Check current state
                state = self.drone.getMultirotorState(drone_name)
                if self.debug:
                    print(f"  {drone_name}: Ready={state.ready}")
                
                # Arm the drone if not already armed
                if not state.ready:
                    if self.debug:
                        print(f"  Arming {drone_name}...")
                    
                    try:
                        # Enable API control first
                        self.drone.enableApiControl(True, vehicle_name=drone_name)
                        
                        # Wait for API control to take effect
                        import time
                        time.sleep(0.2)
                        
                        # Arm the drone
                        self.drone.armDisarm(True, vehicle_name=drone_name)
                        
                        # Takeoff and short hover to engage motors and confirm off ground
                        self.drone.takeoffAsync(vehicle_name=drone_name).join()
                        time.sleep(0.3)
                        self.drone.moveToZAsync(-0.3, 1.0, vehicle_name=drone_name).join()
                        time.sleep(0.3)
                        
                        # Verify arming
                        new_state = self.drone.getMultirotorState(drone_name)
                        is_flying = (new_state.landed_state != airsim.LandedState.Landed)
                        if self.debug:
                            print(f"  {drone_name}: Ready={new_state.ready}, Flying={is_flying}")
                        
                        if new_state.ready or is_flying:
                            print(f"  ✓ {drone_name} armed successfully")
                        else:
                            print(f"  ❌ {drone_name} failed to arm!")
                            # Try one more time
                            self.drone.armDisarm(True, vehicle_name=drone_name)
                            time.sleep(0.5)
                            final_state = self.drone.getMultirotorState(drone_name)
                            if final_state.ready or (final_state.landed_state != airsim.LandedState.Landed):
                                print(f"  ✓ {drone_name} armed on second attempt!")
                            else:
                                print(f"  ❌ {drone_name} still not armed after retry!")
                                
                    except Exception as arm_error:
                        if self.debug:
                            print(f"    Arming error: {arm_error}")
                        print(f"  ❌ {drone_name} failed to arm!")
                        
                else:
                    if self.debug:
                        print(f"  ✓ {drone_name} already armed")
                        
            except Exception as e:
                if self.debug:
                    print(f"  ❌ Error arming {drone_name}: {e}")
        
        if self.debug:
            print("🔧 Drone arming complete")

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def step(self, action):
        self.current_step += 1
        # carry over rewards for terminated agents
        self.reward = {agent: (self.reward[agent] if self.terminations[agent] != 1 else 0) for agent in self.possible_agents}

        # RLlib/Parallel wrappers may include '__all__' in action dict; ignore it
        if isinstance(action, dict) and "__all__" in action:
            action = {k: v for k, v in action.items() if k != "__all__"}

        # Debug: show any survivors missing from the action dict (we'll fallback to last_action)
        if self.debug and isinstance(action, dict):
            missing = [i for i in self.agents if i not in action]
            if missing:
                print("Missing actions for:", missing, "-> using last_action fallback")
        
        # Debug: show action values occasionally
        if self.debug and isinstance(action, dict):
            for agent_id, agent_action in action.items():
                if np.random.random() < 0.05:  # 5% of the time
                    print(f"DEBUG {agent_id} action: {agent_action}")

        # Execute actions for active agents; default to last smoothed action if missing
        for i in self.agents:
            a_i = action.get(i, self._last_action[i])
            self.do_action(a_i, i)
            # Capture learned bids if provided
            try:
                a_arr = np.asarray(a_i, dtype=np.float32)
                if a_arr.shape[0] >= 4 + self.num_tasks:
                    bids = a_arr[4:4 + self.num_tasks]
                    # Temperature shrink to avoid early saturation, then map [-1,1] -> [0,1]
                    b = np.clip(bids, -1.0, 1.0)
                    b = np.clip(b * float(self.bid_temp), -1.0, 1.0)
                    raw = 0.5 * (b + 1.0)
                    # EMA smoothing
                    prev = self._bid_ema.get(i, np.zeros_like(raw))
                    ema = float(self.bid_ema_alpha) * prev + (1.0 - float(self.bid_ema_alpha)) * raw
                    self._bid_ema[i] = ema
                    self._last_bids[i] = ema
            except Exception:
                pass

        obs, info = self.get_obs(self.terminations)
        # Step-level TTC summary (global min over agents if available)
        try:
            ttc_vals = []
            for a in self.agents:
                v = info.get(a, {}).get("ttc_pair_min", None)
                if v is not None:
                    ttc_vals.append(float(v))
            if ttc_vals:
                print(f"GLOBAL TTC_pair_min: {min(ttc_vals):.3f}")
        except Exception:
            pass
        self.reward, self.terminations = self.compute_reward(self.reward, self.terminations, action)

        # PettingZoo agent list update (avoid '__all__')
        self.agents = [k for k in self.possible_agents if self.terminations.get(k, 0) != 1]

        # Always include '__all__' to help RLlib episode accounting
        try:
            self.terminations['__all__'] = all(self.terminations.get(a, 0) == 1 for a in self.possible_agents)
            self.truncations['__all__'] = all(self.truncations.get(a, 0) == 1 for a in self.possible_agents)
        except Exception:
            pass

        if self.debug:
            print("##################################")
            print("########### Step debug ###########")
            print("##################################")
            print(f"Episode idx: {self._episode_index} | Env step: {self.current_step}")
            print("Returned rewards", self.reward)
            print("Active agents (not dead/not success):", self.agents)
            print("Terminated?", self.terminations)
            # Assignment check: each agent's current task
            try:
                assign_map = {aid: int(self._assigned_task.get(aid, -1)) for aid in self.possible_agents}
                print("Assignments (agent->task):", assign_map)
                # Count duplicates across active tasks
                counts = {}
                for t in assign_map.values():
                    if t >= 0:
                        counts[t] = counts.get(t, 0) + 1
                duplicates = {t: c for t, c in counts.items() if c > 1}
                if len(duplicates) > 0:
                    print("⚠️  Multiple agents assigned to the same task:", duplicates)
                unassigned = [aid for aid, t in assign_map.items() if t < 0]
                if len(unassigned) > 0:
                    print("ℹ️  Unassigned agents (defaulting to base):", unassigned)
                # Per-agent distance to assigned base to track progress
                try:
                    dist_map = {}
                    for aid in self.possible_agents:
                        t = int(self._assigned_task.get(aid, -1))
                        if 0 <= t < getattr(self._tasks_pos, 'shape', [0])[0]:
                            base = self._tasks_pos[t]
                        else:
                            base = self._goal_base_pos
                        try:
                            stp = self.drone.getMultirotorState(aid)
                            px = float(stp.kinematics_estimated.position.x_val)
                            py = float(stp.kinematics_estimated.position.y_val)
                            pz = float(stp.kinematics_estimated.position.z_val)
                            d = float(np.linalg.norm(np.array([px, py, pz], dtype=np.float32) - base))
                        except Exception:
                            d = float('nan')
                        dist_map[aid] = round(d, 2)
                    print("Dist-to-assigned-base:", dist_map)
                except Exception:
                    pass
            except Exception:
                pass
        # Filter infos to only current obs keys to satisfy RLlib
        info_filtered = {k: info[k] for k in obs.keys()}
        if self.debug:
            print("Infos?", info_filtered)

        # Return ParallelEnv API tuple
        return obs, self.reward, self.terminations, self.truncations, info_filtered

    def observe(self, agent):
        return self.get_ob(agent_id=agent)

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None,):

        # Apply smoothing warmup for initial episodes
        if self._episode_index < self._smooth_warmup_episodes:
            self.smooth_alpha = self._smooth_warmup_value
        else:
            self.smooth_alpha = 0.10

        self.setup_flight()
        obs, infos = self.get_obs(self.terminations)
        self._episode_index += 1
        return obs, infos

    def render(self):
        return self.get_obs(self.terminations)

    def generate_pos(self):
        # If no spawn_points provided or set to 'here', keep current map positions
        xs, ys, zs = [], [], []
        if (self.spawn_points is None) or (
            isinstance(self.spawn_points, str) and str(self.spawn_points).lower() in ("here", "current", "map")
        ):
            for aid in self.possible_agents:
                try:
                    st = self.drone.getMultirotorState(aid)
                    cx = float(np.clip(st.kinematics_estimated.position.x_val, self.x_min, self.x_max))
                    cy = float(np.clip(st.kinematics_estimated.position.y_val, self.y_min, self.y_max))
                    cz = float(np.clip(st.kinematics_estimated.position.z_val, self.z_min, self.z_max))
                except Exception:
                    cx, cy, cz = 0.0, 0.0, -1.2
                xs.append(cx); ys.append(cy); zs.append(cz)
            return (np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(zs, dtype=np.float32))

        # Exact spawn points provided
        def _get_point(idx, aid):
            sp = self.spawn_points
            if isinstance(sp, dict):
                return sp.get(aid, None)
            try:
                return sp[idx]
            except Exception:
                return None
        for idx, aid in enumerate(self.possible_agents):
            pt = _get_point(idx, aid)
            if pt is None or len(pt) < 3:
                raise ValueError(f"Missing spawn point for agent {aid}")
            cx = float(np.clip(pt[0], self.x_min, self.x_max))
            cy = float(np.clip(pt[1], self.y_min, self.y_max))
            cz = float(np.clip(pt[2], self.z_min, self.z_max))
            xs.append(cx); ys.append(cy); zs.append(cz)
        return (np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(zs, dtype=np.float32))

    # Multi agent start setup
    def setup_flight(self):
        self.drone.reset()

        # Resetting data
        self.reward = {i: 0 for i in self.possible_agents}
        # self.done = {i: 0 for i in self.possible_agents}
        self.terminations = {i: 0 for i in self.possible_agents}
        self.truncations = {i: 0 for i in self.possible_agents}     
        self.obj = {i: 0 for i in self.possible_agents}   
        self.current_step = 0

        # PettingZoo parameters
        self.agents = self.possible_agents[:]

        # For each drone
        for i in self.possible_agents:
            # Reset smoothed action and comm buffers per episode
            self._last_action[i] = np.zeros(4 + self.num_tasks, dtype=np.float32)
            self._comm_prev[i] = np.zeros((self.num, self.comm_dim), dtype=np.float32)
            self._hit_geofence[i] = False

            self.drone.enableApiControl(True, vehicle_name=i)
            
            # Wait for API control to be established
            import time
            time.sleep(0.5)
            
            # Arm the drone
            self.drone.armDisarm(True, vehicle_name=i)
            
            # Wait for arming
            time.sleep(0.5)
            
            # Take off to a lower hover near ground
            self.drone.moveToZAsync(-0.7, 1.0, vehicle_name=i).join()
            # Reset success dwell counters per agent
            self._success_hold[i] = 0

        # Build fixed tasks from scene objects
        self._build_tasks()
        # Capture initially active tasks for episode-completion check
        try:
            self._task_active_init = self._task_mask.copy()
        except Exception:
            self._task_active_init = np.ones((self.num_tasks,), dtype=np.float32)
        # Reset time budget trackers
        self._last_completion_step = 0
        # Backward-compat single target references (use task 0)
        tx, ty, tz = self._tasks_pos[0]
        self.target_pos = np.array([tx, ty])
        self._goal_base_pos = np.array([tx, ty, tz], dtype=np.float32)
        # Choose start x a bit behind the target to reduce wall banging (used for default spawn)
        self.agent_start_pos = float(np.clip(tx - 20.0, self.x_min + 10.0, self.x_max - 10.0))
        # Keep spawn y near target y for shorter initial path (default spawn)
        self._spawn_center_y = ty
      

        # Generate deterministic or constrained spawn positions
        x_pos, y_pos, z_pos = self.generate_pos()
        # De-clump spawn: add small per-agent XY jitter (±1.0 m)
        try:
            rng = np.random.RandomState(seed=int(time.time()) % 1000000)
        except Exception:
            rng = np.random
        jitter_xy = rng.uniform(low=-2.0, high=2.0, size=(len(self.possible_agents), 2)).astype(np.float32)
        x_pos = (x_pos + jitter_xy[:, 0]).astype(np.float32)
        y_pos = (y_pos + jitter_xy[:, 1]).astype(np.float32)
        # Keep fixed ceiling_z = -6.0 per safety policy; do not adapt to spawn altitude
        using_here_spawn = (self.spawn_points is None) or (
            isinstance(self.spawn_points, str) and str(self.spawn_points).lower() in ("here", "current", "map")
        )

        print("Starting y positions:", y_pos)

        # Enforce minimum start–target XY distance only when we actively place spawns
        if not using_here_spawn:
            min_start_target_xy = 5.0
            tx, ty = float(self.target_pos[0]), float(self.target_pos[1])
            for i in range(self.num):
                tries = 0
                while True:
                    dx = float(self.agent_start_pos) - tx
                    dy = float(y_pos[i]) - ty
                    if (dx * dx + dy * dy) ** 0.5 >= min_start_target_xy:
                        break
                    # re-sample y
                    y_pos[i] = float(np.random.uniform(-50, 10))
                    tries += 1
                    if tries > 20:
                        # give up; shift by min distance along y
                        sign = 1.0 if dy >= 0 else -1.0
                        y_pos[i] = ty + sign * min_start_target_xy
                        break

        # Move agents to spawn only when explicit spawn points are provided
        if not using_here_spawn:
            for i in range(0,self.num):
                try:
                    self.drone.moveToPositionAsync(
                        float(x_pos[i]), float(y_pos[i]), float(z_pos[i]),
                        velocity=1.0, vehicle_name=self.possible_agents[i]
                    ).join()
                except Exception:
                    # fallback if backend requires pose set during initialization only
                    pose = airsim.Pose(airsim.Vector3r(float(x_pos[i]), float(y_pos[i]), float(z_pos[i])))
                    self.drone.simSetVehiclePose(pose=pose, ignore_collision=True, vehicle_name=self.possible_agents[i])

        # Removed unused self.target_dist_prev

        if self.input_mode == "multi_rgb":
            self.obs_stack = np.zeros(self.image_shape)

        # Get collision time stamp    
        self.collision_time = {i: self.drone.simGetCollisionInfo(vehicle_name=i).time_stamp for i in self.possible_agents}

        # Initialize per-agent previous distance to target and idle counters
        self._prev_target_dist = {}
        # Track progress to task center (base) separately from micro-formation desired
        self._prev_goal_dist_base = {}
        self._initial_goal_dist_base = {}
        self._idle_steps = {}
        self._away_steps = {}
        self._prev_pos = {}
        self._desired_pos = {}
        self._initial_target_dist = {}
        # Disable body-frame long-dt warmup steps; train straight without special debug steps
        self._bf_debug_steps_remaining = {i: 0 for i in self.possible_agents}
        # Compute initial allocation and desired positions
        pos_init_all = {}
        for i in self.possible_agents:
            try:
                st = self.drone.getMultirotorState(i)
                x_i, y_i, z_i = (
                    float(st.kinematics_estimated.position.x_val),
                    float(st.kinematics_estimated.position.y_val),
                    float(st.kinematics_estimated.position.z_val),
                )
            except Exception:
                x_i, y_i, z_i = 0.0, 0.0, 0.0
            pos_init_all[i] = (x_i, y_i, z_i)
        # Update allocation and set desired positions accordingly
        self._update_allocation(pos_init_all)
        for i in self.possible_agents:
            # If allocation didn't set a desired pos (unlikely), default to task 0 + static offset
            if i not in self._desired_pos:
                self._desired_pos[i] = (self._goal_base_pos + self.agent_offsets[i]).astype(np.float32)
            # Previous 3D error
            d_init = float(np.linalg.norm(np.array([x_i, y_i, z_i]) - self._desired_pos[i]))
            self._prev_target_dist[i] = d_init
            self._initial_target_dist[i] = d_init
            # Base distance to assigned task center (fallback to global base if unknown)
            try:
                t_i = int(self._assigned_task.get(i, -1))
                base = self._tasks_pos[t_i] if 0 <= t_i < self._tasks_pos.shape[0] else self._goal_base_pos
            except Exception:
                base = self._goal_base_pos
            d_init_base = float(np.linalg.norm(np.array([x_i, y_i, z_i]) - base))
            self._prev_goal_dist_base[i] = d_init_base
            self._initial_goal_dist_base[i] = d_init_base
            self._idle_steps[i] = 0
            self._away_steps[i] = 0
            self._prev_pos[i] = np.array([x_i, y_i, z_i], dtype=np.float32)


    def do_action(self, action, name):
        # Apply action smoothing on control channels only (exclude bid dims)
        prev = self._last_action[name]
        act = np.asarray(action, dtype=np.float32)
        act = np.clip(act, -1.0, 1.0)
        ctrl_prev = prev[:4]
        ctrl_act = act[:4]
        smoothed_ctrl = self.smooth_alpha * ctrl_prev + (1.0 - self.smooth_alpha) * ctrl_act
        self._last_action[name] = np.concatenate([smoothed_ctrl, act[4:]], axis=0)
        

        # Map to physical commands (no extra velocity low-pass)
        spd_scale = float(self.agent_speed_scale.get(name, 1.0))
        vx = float(smoothed_ctrl[1] * self.max_xy_speed * spd_scale)   # forward/back in body frame
        vy = float(smoothed_ctrl[0] * self.max_xy_speed * spd_scale)   # left/right in body frame
        # AirSim NED: positive z is downward; positive action[2] should mean up -> invert sign
        vz = float(-smoothed_ctrl[2] * self.max_z_speed)
        
        # Removed lateral sidestep/braking to avoid sideways nudges; rely on spacing penalty only
        # Zero yaw during first few episodes to avoid spirals
        if self._episode_index < 5:
            yaw_rate = 0.0
        else:
            # Keep yaw near zero when far; allow limited yaw near goal
            try:
                t_i = int(self._assigned_task.get(name, -1))
                base = self._tasks_pos[t_i] if 0 <= t_i < self._tasks_pos.shape[0] else self._goal_base_pos
                st_self = self.drone.getMultirotorState(name)
                p_self = np.array([
                    float(st_self.kinematics_estimated.position.x_val),
                    float(st_self.kinematics_estimated.position.y_val),
                    float(st_self.kinematics_estimated.position.z_val),
                ], dtype=np.float32)
                base_dist_curr = float(np.linalg.norm(p_self - base))
            except Exception:
                base_dist_curr = 1e9
            if base_dist_curr > float(self.form_gate_enable_m):
                yaw_rate = 0.0
            else:
                yaw_rate = float(smoothed_ctrl[3] * (0.5 * self.max_yaw_rate_deg))

        # Optional latency
        # No artificial latency in fast sim mode

        # Execute velocity and yaw concurrently for control_dt
        try:
            # Measure BEFORE, then command, then measure AFTER over the same duration
            st_pre = self.drone.getMultirotorState(name)
            pos_before = np.array([
                float(st_pre.kinematics_estimated.position.x_val),
                float(st_pre.kinematics_estimated.position.y_val),
                float(st_pre.kinematics_estimated.position.z_val),
            ])

            # Enforce single ceiling: block further ascent if already above ceiling
            try:
                if float(pos_before[2]) < float(self.ceiling_z) and float(vz) < 0.0:
                    vz = 0.0
                # Note: No extra descent forcing; rely on reward penalty
            except Exception:
                pass

            # For the first N steps use longer dt and zero/low yaw in one combined BODY-frame command
            if self._bf_debug_steps_remaining.get(name, 0) > 0:
                dt_debug = 3.0
                f1 = self.drone.moveByVelocityBodyFrameAsync(
                    vx, vy, vz,
                    duration=dt_debug,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                    vehicle_name=name,
                )
                f1.join()
                effective_dt = dt_debug
                self._bf_debug_steps_remaining[name] = self._bf_debug_steps_remaining.get(name, 0) - 1
            else:
                f1 = self.drone.moveByVelocityBodyFrameAsync(
                    vx, vy, vz,
                    duration=self.control_dt,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
                    vehicle_name=name,
                )
                f1.join()
                effective_dt = float(self.control_dt)

            if self.debug:
                st_post = self.drone.getMultirotorState(name)
                pos_after = np.array([
                    float(st_post.kinematics_estimated.position.x_val),
                    float(st_post.kinematics_estimated.position.y_val),
                    float(st_post.kinematics_estimated.position.z_val),
                ])

                movement = np.linalg.norm(pos_after - pos_before)
                # Rotate body-frame velocity into NED to compute apples-to-apples expected displacement
                q = st_pre.kinematics_estimated.orientation
                w, xq, yq, zq = float(q.w_val), float(q.x_val), float(q.y_val), float(q.z_val)
                R = np.array([
                    [1 - 2*(yq*yq + zq*zq),     2*(xq*yq - zq*w),     2*(xq*zq + yq*w)],
                    [    2*(xq*yq + zq*w), 1 - 2*(xq*xq + zq*zq),     2*(yq*zq - xq*w)],
                    [    2*(xq*zq - yq*w),     2*(yq*zq + xq*w), 1 - 2*(xq*xq + yq*yq)],
                ], dtype=np.float32)
                v_body = np.array([vx, vy, vz], dtype=np.float32)
                v_ned = R @ v_body
                expected_movement = float(np.linalg.norm(v_ned)) * float(effective_dt)

             
                print(f"DEBUG {name}: Actual movement: {movement:.6f}, Expected: {expected_movement:.6f}")

                if movement < 0.001 and expected_movement > 0.01:
                    if name in self.agents and self.terminations.get(name, 0) != 1:
                        print(f"⚠️  WARNING {name}: Commands sent but NO MOVEMENT detected!")
                        print(f"  This suggests AirSim commands are failing silently!")
                        print(f"  Drone status: Active, Ready={self.drone.getMultirotorState(name).ready}")
                    else:
                        print(f"ℹ️  {name}: No movement expected (drone terminated or inactive)")
                
        except Exception as e:
            # Fallback without yaw control
            if self.debug:
                print(f"ERROR {name}: Command failed: {e}")
            try:
                self.drone.moveByVelocityBodyFrameAsync(vx, vy, vz, duration=self.control_dt, vehicle_name=name).join()
            except Exception as e2:
                if self.debug:
                    print(f"ERROR {name}: Fallback command also failed: {e2}")
                    print(f"  CRITICAL: AirSim is not responding to any commands!")

        # Post-action geofence check in NED; avoid UE-world teleports
        try:
            st = self.drone.getMultirotorState(name)
            px = float(st.kinematics_estimated.position.x_val)
            py = float(st.kinematics_estimated.position.y_val)
            pz = float(st.kinematics_estimated.position.z_val)
            clamp_x = float(np.clip(px, self.x_min, self.x_max))
            clamp_y = float(np.clip(py, self.y_min, self.y_max))
            clamp_z = pz  # do not clamp Z during debugging
            if (abs(clamp_x - px) > 1e-3) or (abs(clamp_y - py) > 1e-3) or (abs(clamp_z - pz) > 1e-3):
                # Mark geofence and gently nudge back using NED velocity
                self._last_action[name] *= 0.3
                self._hit_geofence[name] = True
                nx = -0.3 if clamp_x != px and px > clamp_x else (0.3 if clamp_x != px and px < clamp_x else 0.0)
                ny = -0.3 if clamp_y != py and py > clamp_y else (0.3 if clamp_y != py and py < clamp_y else 0.0)
                # Keep altitude unchanged for nudge
                try:
                    self.drone.moveByVelocityAsync(nx, ny, 0.0, duration=0.2, vehicle_name=name).join()
                except Exception:
                    pass
                # If altitude is out of bounds, correct it explicitly to avoid repeated clamps
                if abs(clamp_z - pz) > 1e-3:
                    try:
                        self.drone.moveToZAsync(clamp_z, 1.0, vehicle_name=name).join()
                    except Exception:
                        pass
        except Exception:
            pass

        # REMOVED: This was causing vibration and hovering!
        # self.drone.moveByVelocityAsync(vx=3, vy=3, vz=0, duration=1, vehicle_name=name)

    # Multi agent observations as list of single obs
    def get_obs(self, terminations):
        obs = {}
        local_info = {}
        # First collect positions for all agents (fixed order) to build team_pos of shape (self.num, 3)
        pos_all = {}
        # Reset depth cache for this step
        self._depth_cache = {}
        for agent_id in self.possible_agents:
            try:
                st = self.drone.getMultirotorState(agent_id)
                x, y, z = (
                    float(st.kinematics_estimated.position.x_val),
                    float(st.kinematics_estimated.position.y_val),
                    float(st.kinematics_estimated.position.z_val),
                )
                pos_all[agent_id] = (x, y, z)
            except Exception:
                pos_all[agent_id] = (0.0, 0.0, 0.0)
            # Pre-fetch and cache depth once per agent per step
            try:
                d = self.get_depth_image(thresh=1.5, name=agent_id)
                self._depth_cache[agent_id] = d
            except Exception:
                self._depth_cache[agent_id] = None

        team_pos_full = np.array([pos_all[aid] for aid in self.possible_agents], dtype=np.float32)

        # Update tasks/allocation and desired positions for this step
        self._update_allocation(pos_all)
        # Compute per-task counts and coalition filled flags
        try:
            assigned_counts = [0 for _ in range(self.num_tasks)]
            for aid, t in self._assigned_task.items():
                if 0 <= int(t) < self.num_tasks:
                    assigned_counts[int(t)] += 1
            coalition_filled = [
                1.0 if assigned_counts[t] >= (self.task_required[t] if t < len(self.task_required) else 0) else 0.0
                for t in range(self.num_tasks)
            ]
        except Exception:
            assigned_counts = [0 for _ in range(self.num_tasks)]
            coalition_filled = [0.0 for _ in range(self.num_tasks)]

        # Build comm packets with noise/dropout/delay per receiver (agent i)
        # Base packets from true positions and collision flags
        base_packets = []
        base_valid = []
        for aid in self.possible_agents:
            coll = 1.0 if self.is_collision(aid) else 0.0
            px, py, pz = pos_all[aid]
            pkt = np.array([px, py, pz, coll], dtype=np.float32)
            base_packets.append(pkt)
            base_valid.append(1.0)
        base_packets = np.stack(base_packets, axis=0)
        base_valid = np.array(base_valid, dtype=np.float32)

        for i in self.agents:
            local_info[i] = {"collision": self.is_collision(i)}
            x, y, z = pos_all[i]

            if self.input_mode == "multi_rgb":
                obs_t = self.get_rgb_image()	
                obs_t_gray = cv2.cvtColor(obs_t, cv2.COLOR_BGR2GRAY)
                self.obs_stack[:, :, 0] = self.obs_stack[:, :, 1]
                self.obs_stack[:, :, 1] = self.obs_stack[:, :, 2]
                self.obs_stack[:, :, 2] = obs_t_gray
                obs_img = np.hstack((
                    self.obs_stack[:, :, 0],
                    self.obs_stack[:, :, 1],
                    self.obs_stack[:, :, 2]))
                obs_img = np.expand_dims(obs_img, axis=2)
                # NOTE: Not returning multi_rgb dict for now
            elif self.input_mode == "single_rgb":
                obs[i] = {}
                idx_norm = float(self.possible_agents.index(i)) / max(1, (self.num - 1))
                if terminations[i] != 1:
                    img = self.get_rgb_image(i)
                    # ensure shape and dtype
                    if img.shape != self.image_shape:
                        img = cv2.resize(img, (self.image_shape[1], self.image_shape[0])).reshape(self.image_shape)

                    # Keep RGB as-is; remove fused depth
                    obs[i]['cam'] = img.astype(np.uint8)
                    pos_vec = np.array([x, y, z], dtype=np.float32)
                    # Inject small GPS-like noise if enabled (keeps same shape)
                    if self.use_noisy_pos:
                        try:
                            noise = self.np_random.normal(0.0, self.gps_noise_std, size=(3,)).astype(np.float32)
                            pos_vec = pos_vec + noise
                        except Exception:
                            pass
                    pos_low = self.observation_spaces[i]["pos"].low
                    pos_high = self.observation_spaces[i]["pos"].high
                    obs[i]['pos'] = np.clip(pos_vec, pos_low, pos_high)
                    # LiDAR: 32-bin polar min-range around forward axis
                    try:
                        pts = self.get_lidar_points(i)
                        if pts.shape[0] >= 1:
                            # Compute polar in body frame (assumed)
                            angles = np.arctan2(pts[:,1], pts[:,0])  # [-pi, pi]
                            ranges = np.sqrt(np.maximum(1e-6, pts[:,0]**2 + pts[:,1]**2))
                            num_bins = 32
                            bins = np.linspace(-np.pi, np.pi, num_bins + 1)
                            lidar_bins = np.full((num_bins,), 1000.0, dtype=np.float32)
                            idxs = np.digitize(angles, bins) - 1
                            idxs = np.clip(idxs, 0, num_bins - 1)
                            for bi in range(num_bins):
                                mask_bin = (idxs == bi)
                                if np.any(mask_bin):
                                    lidar_bins[bi] = float(np.minimum(lidar_bins[bi], np.min(ranges[mask_bin])))
                        else:
                            lidar_bins = np.full((32,), 1000.0, dtype=np.float32)
                    except Exception:
                        lidar_bins = np.full((32,), 1000.0, dtype=np.float32)
                    # Normalize and cap LiDAR: clamp to [0, lidar_cap_m] then map to 0..1 with near=1
                    try:
                        cap = float(self.lidar_cap_m)
                    except Exception:
                        cap = 12.0
                    if self.disable_lidar:
                        obs[i]['lidar'] = np.zeros((32,), dtype=np.float32)
                    else:
                        lidar_clipped = np.clip(lidar_bins, 0.0, cap)
                        lidar_norm = 1.0 - (lidar_clipped / max(1e-6, cap))
                        obs[i]['lidar'] = lidar_norm.astype(np.float32)
                    # TTC scalars
                    # Pairwise TTC (min over peers) from positions and finite-diff velocity
                    try:
                        # Self velocity (finite diff)
                        prev = self._prev_pos.get(i, np.array([x, y, z], dtype=np.float32))
                        vel_i = (np.array([x, y, z], dtype=np.float32) - prev) / max(self.control_dt, 1e-6)
                        ttc_pair = 1e6
                        for j in self.possible_agents:
                            if j == i:
                                continue
                            pj = pos_all.get(j, None)
                            if pj is None:
                                continue
                            r = np.array([pj[0]-x, pj[1]-y, pj[2]-z], dtype=np.float32)
                            # Approx peer velocity as finite diff too
                            prev_j = self._prev_pos.get(j, np.array([pj[0], pj[1], pj[2]], dtype=np.float32))
                            vel_j = (np.array([pj[0], pj[1], pj[2]], dtype=np.float32) - prev_j) / max(self.control_dt, 1e-6)
                            v = vel_j - vel_i
                            r_norm = float(np.linalg.norm(r) + 1e-6)
                            closing = float(-(r @ v) / r_norm)
                            if closing > 0:
                                ttc = r_norm / closing
                                if ttc < ttc_pair:
                                    ttc_pair = ttc
                    except Exception:
                        ttc_pair = 1e6
                    # Cache for reward penalty later
                    self._last_ttc_pair[i] = float(ttc_pair)
                    # Velocity-aligned TTC from LiDAR bins (choose bin closest to velocity heading)
                    try:
                        # Compute body-frame planar velocity from last action
                        last = self._last_action.get(i, np.zeros(4, dtype=np.float32))
                        vx_body = float(last[1] * self.max_xy_speed)
                        vy_body = float(last[0] * self.max_xy_speed)
                        speed = float(np.hypot(vx_body, vy_body))
                        if speed > 0.05:
                            # Heading angle in body frame [-pi, pi]
                            heading = float(np.arctan2(vy_body, vx_body))
                            # Map to LiDAR bins (centered around forward)
                            num_bins = 32
                            bins = np.linspace(-np.pi, np.pi, num_bins + 1)
                            # Select 3-bin neighborhood around heading for robustness
                            bin_center = int(np.clip(np.digitize(heading, bins) - 1, 0, num_bins - 1))
                            idxs = [(bin_center + k) % num_bins for k in (-1, 0, 1)]
                            neigh = np.array([lidar_bins[j] for j in idxs], dtype=np.float32)
                            neigh[neigh <= 0.0] = 1000.0
                            d_heading = float(np.min(neigh))
                            ttc_any = d_heading / speed
                        else:
                            d_heading = 1e6
                            ttc_any = 1e6
                        # store ranges for reward penalty
                        self._last_forward_range[i] = float(lidar_bins[16]) if 0 <= 16 < 32 else d_heading
                        self._last_heading_range[i] = d_heading
                    except Exception:
                        ttc_any = 1e6
                        self._last_forward_range[i] = 1e6
                        self._last_heading_range[i] = 1e6
                    # Obs ttc: [ttc_pair_min, ttc_velocity_aligned]
                    obs[i]['ttc'] = np.array([ttc_pair, ttc_any], dtype=np.float32)
                    # Depth not part of observation space; keep cached for reward only
                    # Legacy team_pos (still filled)
                    tp_low = self.observation_spaces[i]["team_pos"].low
                    tp_high = self.observation_spaces[i]["team_pos"].high
                    obs[i]['team_pos'] = np.clip(team_pos_full, tp_low, tp_high)
                    # Tasks
                    obs[i]['task_pos'] = self._tasks_pos.astype(np.float32)
                    obs[i]['task_mask'] = self._task_mask.astype(np.float32)
                    # Infos for logging/metrics
                    try:
                        chosen_t = int(self._assigned_task.get(i, -1))
                        commit = float(self._intent_fields.get(i, np.zeros(4, dtype=np.float32))[3])
                        bids = self._last_bids.get(i, np.zeros((self.num_tasks,), dtype=np.float32))
                        bid_self = float(bids[chosen_t]) if 0 <= chosen_t < self.num_tasks else 0.0
                        desired = self._desired_pos.get(i, self._goal_base_pos)
                        form_err = float(np.linalg.norm(np.array([x, y, z], dtype=np.float32) - desired))
                        local_info[i]["chosen_task"] = float(chosen_t)
                        local_info[i]["commit"] = float(commit)
                        local_info[i]["bid_self"] = float(bid_self)
                        # Movement diagnostics: distances to current desired slot and task base
                        try:
                            if 0 <= chosen_t < self._tasks_pos.shape[0]:
                                base_pos = self._tasks_pos[chosen_t]
                            else:
                                base_pos = self._goal_base_pos
                            dist_to_base = float(np.linalg.norm(np.array([x, y, z], dtype=np.float32) - base_pos))
                        except Exception:
                            base_pos = self._goal_base_pos
                            dist_to_base = float(np.linalg.norm(np.array([x, y, z], dtype=np.float32) - base_pos))
                        local_info[i]["dist_to_desired"] = form_err
                        local_info[i]["dist_to_base"] = dist_to_base
                        # Global per-task counts and requirements
                        for t in range(self.num_tasks):
                            local_info[i][f"assigned_count_task{t}"] = float(assigned_counts[t])
                            req = float(self.task_required[t] if t < len(self.task_required) else 0)
                            local_info[i][f"r_t_task{t}"] = req
                            local_info[i][f"coalition_filled_task{t}"] = float(coalition_filled[t])
                        # Safety/formation
                        local_info[i]["ttc_pair_min"] = float(ttc_pair)
                        # Provide velocity-aligned TTC only (forward-only deprecated)
                        try:
                            local_info[i]["ttc_heading"] = float(ttc_any)
                        except Exception:
                            # fallback if local variable not in scope
                            ttc_val = 1e6
                            try:
                                ttcs = obs.get(i, {}).get("ttc", None)
                                if ttcs is not None and len(ttcs) >= 2:
                                    ttc_val = float(ttcs[1])
                            except Exception:
                                pass
                            local_info[i]["ttc_heading"] = ttc_val
                        # Is the agent moving toward the current task base this step?
                        try:
                            prev_base = float(self._prev_goal_dist_base.get(i, dist_to_base))
                            local_info[i]["moving_to_base"] = 1.0 if (prev_base - dist_to_base) > 0.1 else 0.0
                        except Exception:
                            local_info[i]["moving_to_base"] = 0.0
                        # Report reallocation source once, then clear mark
                        try:
                            mark = self._realloc_mark.get(i, None)
                            if mark is not None:
                                local_info[i]["realloc_from"] = float(mark[0])
                                self._realloc_mark[i] = None
                        except Exception:
                            pass
                    except Exception:
                        pass

                    # Comm packets with noise/drop/delay (receiver-specific, relative positions)
                    packets = base_packets.copy()
                    mask = base_valid.copy()
                    # Add Gaussian noise to positions
                    packets[:, :3] += np.random.normal(0.0, self.comm_noise_std, size=(self.num, 3)).astype(np.float32)
                    # Dropout
                    drop = np.random.rand(self.num) < self.comm_drop_prob
                    mask[drop] = 0.0
                    # Delay: use previous packet for some peers
                    delay = np.random.rand(self.num) < self.comm_delay_prob
                    if np.any(delay):
                        prev_pkt = self._comm_prev.get(i, np.zeros_like(packets))
                        packets[delay] = prev_pkt[delay]
                    # Convert positions to relative (peer_pos - self_pos)
                    packets[:, 0] -= x
                    packets[:, 1] -= y
                    packets[:, 2] -= z
                    # Mask out self entry so attention doesn't collapse onto self
                    self_idx = self.possible_agents.index(i)
                    if 0 <= self_idx < self.num:
                        mask[self_idx] = 0.0
                    # Append intent fields per sender: [task_id_norm, bid, eta, commit]
                    intent = np.zeros((self.num, 4), dtype=np.float32)
                    for idx_sender, aid in enumerate(self.possible_agents):
                        fields = self._intent_fields.get(aid, np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                        intent[idx_sender] = fields
                    packets_ext = np.concatenate([packets, intent], axis=1)
                    # No-comms mode: zero out packets and mask
                    if self.comm_mode == "none":
                        packets_ext[:] = 0.0
                        mask[:] = 0.0
                    # Save for next time (per receiver)
                    self._comm_prev[i] = packets.copy()
                    self._comm_prev_mask[i] = mask.copy()
                    # Clip and assign unified team_comm
                    tc_low = self.observation_spaces[i]["team_comm"].low
                    tc_high = self.observation_spaces[i]["team_comm"].high
                    obs[i]['team_comm'] = np.clip(packets_ext, tc_low, tc_high)
                    obs[i]['team_comm_mask'] = mask.astype(np.float32)
                    obs[i]['agent_idx'] = np.array([idx_norm], dtype=np.float32)
                    # Remaining time in observation
                    try:
                        rt = max(0.0, 1.0 - float(self.current_step or 0) / max(1.0, float(self.global_time_budget_steps)))
                    except Exception:
                        rt = 0.0
                    obs[i]['remaining_time'] = np.array([rt], dtype=np.float32)
                else:
                    obs[i]['cam'] = np.zeros((self.image_shape), dtype=np.uint8)
                    obs[i]['pos'] = np.zeros((3,), dtype=np.float32)
                    obs[i]['lidar'] = np.zeros((32,), dtype=np.float32)
                    tp_low = self.observation_spaces[i]["team_pos"].low
                    tp_high = self.observation_spaces[i]["team_pos"].high
                    obs[i]['team_pos'] = np.clip(team_pos_full, tp_low, tp_high)
                    obs[i]['task_pos'] = self._tasks_pos.astype(np.float32)
                    obs[i]['task_mask'] = self._task_mask.astype(np.float32)
                    # Comm zeroed on termination
                    obs[i]['team_comm'] = np.zeros((self.num, 8), dtype=np.float32)
                    obs[i]['team_comm_mask'] = np.zeros((self.num,), dtype=np.float32)
                    obs[i]['agent_idx'] = np.array([idx_norm], dtype=np.float32)
                    obs[i]['remaining_time'] = np.array([0.0], dtype=np.float32)
            elif self.input_mode == "depth":
                depth = self.get_depth_image(thresh=3.4).reshape(self.image_shape)
                depth = ((depth/3.4)*255).astype(int)

              
        return obs, local_info

    def _build_tasks(self):
        # Fixed tasks from configured scene object names
        names = list(self.task_names)
        pos_list = []
        mask = []
        for name in names:
            try:
                pose = self.drone.simGetObjectPose(name)
                px, py, pz = float(pose.position.x_val), float(pose.position.y_val), float(pose.position.z_val)
                # Use scene object's Z without ceiling clamping; single ceiling logic handled in control/reward
                px = float(np.clip(px, self.x_min + 5.0, self.x_max - 5.0))
                py = float(np.clip(py, self.y_min + 5.0, self.y_max - 5.0))
                pos_list.append((px, py, pz))
                mask.append(1.0)
            except Exception:
                # Task not found; keep placeholder and mark invalid
                pos_list.append((0.0, 0.0, -2.0))
                mask.append(0.0)
        # Ensure shape (num_tasks,3)
        if len(pos_list) < self.num_tasks:
            pos_list += [(0.0, 0.0, -2.0)] * (self.num_tasks - len(pos_list))
            mask += [0.0] * (self.num_tasks - len(mask))
        self._tasks_pos = np.array(pos_list[: self.num_tasks], dtype=np.float32)
        self._task_mask = np.array(mask[: self.num_tasks], dtype=np.float32)

    def _update_allocation(self, pos_all: dict):
        # Value-aware, queue-based learned selection
        K = self._tasks_pos.shape[0]
        if K < 1:
            self._build_tasks()
            K = self._tasks_pos.shape[0]
        caps = [1 for _ in range(K)]

        # Previous assignment snapshot for active agents
        prev_assign = {aid: int(self._assigned_task_prev.get(aid, -1)) for aid in self.agents}
        prev_counts = [0 for _ in range(K)]
        for aid, t in prev_assign.items():
            if 0 <= t < K:
                prev_counts[t] += 1

        # Determine under-filled tasks based on SUCCESS counts (exclude completed and masked)
        try:
            unfilled = [
                t for t in range(K)
                if (self._task_mask[t] > 0.5)
                and (not bool(self._task_completed[t]))
                and (int(self._task_success_count[t]) < int(caps[t]))
            ]
        except Exception:
            unfilled = [t for t in range(K) if (not bool(self._task_completed[t]))]

        # Maintain persistent FIFO task queue
        # Add new unfilled tasks
        for t in unfilled:
            if t not in self._task_queue:
                self._task_queue.append(t)
        # Remove tasks that are no longer unfilled
        self._task_queue = [t for t in self._task_queue if t in unfilled]

        # Compute max value among currently queued tasks
        if len(self._task_queue) > 0:
            try:
                max_unfilled_val = max(float(self.task_value[t]) for t in self._task_queue)
            except Exception:
                max_unfilled_val = 0.0
        else:
            max_unfilled_val = 0.0

        # Helper: get bids (with lock bias) for an agent
        def get_bids_for_agent(aid: str) -> np.ndarray:
            bids = self._last_bids.get(aid, np.zeros((K,), dtype=np.float32)).copy()
            lock_until = int(self._commit_lock_until.get(aid, 0))
            cur_t = int(self._assigned_task_prev.get(aid, -1))
            # HARD STICKINESS: if agent has a current task that is not completed yet,
            # force bidding to that task only until success/crash
            try:
                if 0 <= cur_t < K and (not bool(self._task_completed[cur_t])) and (cur_t not in self._agent_completed_tasks.get(aid, set())):
                    forced = np.zeros_like(bids)
                    forced[cur_t] = 1.0
                    return forced
            except Exception:
                pass
            if self.current_step is not None and self.current_step < lock_until and 0 <= cur_t < K:
                bids *= 0.5
                bids[cur_t] = max(bids[cur_t], 1.0)
            return bids

        assigned: dict = {}
        task_counts = [0 for _ in range(K)]

        # Pass 1: HARD KEEP — keep agents on their current task while it remains incomplete
        for aid in self.agents:
            cur_t = prev_assign.get(aid, -1)
            lock_until = int(self._commit_lock_until.get(aid, 0))
            # Keep if current task exists, is not marked completed, and capacity allows
            try:
                current_incomplete = (0 <= cur_t < K) and (not bool(self._task_completed[cur_t]))
            except Exception:
                current_incomplete = (0 <= cur_t < K)
            if current_incomplete and task_counts[cur_t] < caps[cur_t]:
                assigned[aid] = (cur_t, 0.0, 1.0)
                task_counts[cur_t] += 1

        # Pass 2: Assign remaining agents to queued (unfilled) tasks in FIFO order, but only if moving to a strictly higher-value task
        remaining = [aid for aid in self.agents if aid not in assigned]
        for t in list(self._task_queue):
            while task_counts[t] < caps[t]:
                # pick best bidder among remaining who would move up in value and is not locked
                best_aid = None
                best_score = -1.0
                for aid in remaining:
                    cur_t = prev_assign.get(aid, -1)
                    cur_val = float(self.task_value[cur_t]) if 0 <= cur_t < K else -1e9
                    # require strict upward move in value
                    if cur_val >= float(self.task_value[t]):
                        continue
                    # skip locked agents
                    lock_until = int(self._commit_lock_until.get(aid, 0))
                    if self.current_step is not None and self.current_step < lock_until:
                        continue
                    bids = get_bids_for_agent(aid)
                    score = float(bids[t]) if 0 <= t < K else 0.0
                    if score > best_score:
                        best_score = score
                        best_aid = aid
                if best_aid is None:
                    break
                assigned[best_aid] = (t, 0.0, 1.0)
                task_counts[t] += 1
                remaining = [r for r in remaining if r != best_aid]

        # Pass 3: Fallback – keep agents on current if capacity allows; else assign by best bid to any task (commit=0)
        for aid in list(remaining):
            cur_t = prev_assign.get(aid, -1)
            if 0 <= cur_t < K and task_counts[cur_t] < caps[cur_t]:
                assigned[aid] = (cur_t, 0.0, 1.0)
                task_counts[cur_t] += 1
                remaining.remove(aid)

        for aid in list(remaining):
            bids = get_bids_for_agent(aid)
            # choose best feasible task that is not yet filled (caps=1)
            order = list(np.argsort(-bids)) if K > 0 else [0]
            sel_t = None
            for t in order:
                if task_counts[t] < caps[t]:
                    sel_t = int(t)
                    break
            if sel_t is not None:
                assigned[aid] = (sel_t, 0.0, 0.0)  # not committed preference
                task_counts[sel_t] += 1

        # Save intent fields and desired positions (micro-formation)
        team_lists = {t: [] for t in range(K)}
        for aid, (t, eta, commit) in assigned.items():
            team_lists[t].append(aid)
        for t in range(K):
            team_lists[t].sort(key=lambda s: int(s.replace('drone', '')) if s.startswith('drone') else s)

        self._assigned_task = {aid: -1 for aid in self.possible_agents}
        for t in range(K):
            base = self._tasks_pos[t]
            members = team_lists[t]
            for aid in members:
                # No formation: desired is the task center
                desired = np.array(base, dtype=np.float32)
                # Respect single ceiling only in control/reward; do not clamp desired Z here
                desired[2] = float(desired[2])
                self._desired_pos[aid] = desired
                self._assigned_task[aid] = t

        # Print reallocations (only for active agents) and set dwell windows
        try:
            for aid in self.agents:
                prev_t = int(self._assigned_task_prev.get(aid, -1))
                new_t = int(self._assigned_task.get(aid, -1))
                if prev_t != -1 and new_t != -1 and prev_t != new_t:
                    # Only reassignments that go to higher-value tasks are allowed above
                    print(f"Realloc (value-aware): {aid} {prev_t} -> {new_t}")
                    # If agent is currently landed, auto-takeoff for new assignment
                    try:
                        st_now = self.drone.getMultirotorState(aid)
                        if st_now.landed_state == airsim.LandedState.Landed:
                            self.drone.takeoffAsync(vehicle_name=aid).join()
                            # small hover
                            self.drone.moveToZAsync(-0.7, 1.0, vehicle_name=aid).join()
                    except Exception:
                        pass
                    try:
                        p = np.array(self.drone.getMultirotorState(aid).kinematics_estimated.position.as_numpy_array(), dtype=np.float32)
                        dist = float(np.linalg.norm(self._tasks_pos[new_t] - p))
                    except Exception:
                        dist = 20.0
                    horiz_speed = max(0.5, float(self.max_xy_speed) * float(self.agent_speed_scale.get(aid, 1.0)))
                    steps_needed = int(np.ceil((dist / horiz_speed) / max(1e-3, float(self.control_dt))))
                    dwell = int(np.clip(steps_needed, 30, 60))
                    self._commit_lock_until[aid] = int(self.current_step or 0) + dwell
            self._assigned_task_prev = dict(self._assigned_task)
        except Exception:
            pass

        # Intent fields per agent (sender): [task_id_norm, bid, eta, commit]
        denom = float(max(1, K - 1))
        for aid, (t, eta, commit) in assigned.items():
            tid_norm = float(t) / denom if K > 1 else 0.0
            bids = self._last_bids.get(aid, np.zeros((K,), dtype=np.float32))
            bid_val = float(bids[t]) if 0 <= t < K else 0.0
            try:
                px, py, pz = pos_all.get(aid, (0.0, 0.0, 0.0))
                p = np.array([px, py, pz], dtype=np.float32)
                dist = float(np.linalg.norm(self._tasks_pos[t] - p)) if 0 <= t < K else 0.0
                spd = float(self.max_xy_speed) * float(self.agent_speed_scale.get(aid, 1.0))
                eta_val = dist / max(1e-6, spd)
            except Exception:
                eta_val = 0.0
            self._intent_fields[aid] = np.array([tid_norm, bid_val, float(eta_val), float(commit)], dtype=np.float32)

    # Multi agent reward
    def compute_reward(self, reward, terminations, act):
        coord = {}

        if self.current_step >= self.max_steps:
            # GENTLE timeout penalty per agent (reduced from -100 to -20)
            reward = {}
            for i in self.possible_agents:
                try:
                    pos_obj = self.drone.simGetVehiclePose(i).position
                    x_i, y_i, z_i = float(pos_obj.x_val), float(pos_obj.y_val), float(pos_obj.z_val)
                except Exception:
                    x_i, y_i, z_i = 0.0, 0.0, 0.0
                desired = self._desired_pos.get(i, self._goal_base_pos)
                curr = float(np.linalg.norm(np.array([x_i, y_i, z_i]) - desired))
                init = float(max(1e-6, self._initial_target_dist.get(i, curr)))
                reward[i] = -20.0 * (curr / init)  # Much gentler timeout penalty
            terminations = {i: 1 for i in self.possible_agents}
            self.truncations = {i: 1 for i in self.possible_agents}     
            self.obj = {i: -1 for i in self.possible_agents}
            return reward, terminations

        # Precompute current 3D distances for all active agents
        curr_dist = {}
        base_dist = {}
        for i in self.agents:
            st = self.drone.getMultirotorState(i)
            x_i, y_i, z_i = (
                float(st.kinematics_estimated.position.x_val),
                float(st.kinematics_estimated.position.y_val),
                float(st.kinematics_estimated.position.z_val),
            )
            desired = self._desired_pos.get(i, self._goal_base_pos)
            d = float(np.linalg.norm(np.array([x_i, y_i, z_i]) - desired))
            curr_dist[i] = d
            # Distance to task center (base) ignoring micro-formation offsets
            try:
                t_i = int(self._assigned_task.get(i, -1))
                base = self._tasks_pos[t_i] if 0 <= t_i < self._tasks_pos.shape[0] else self._goal_base_pos
            except Exception:
                base = self._goal_base_pos
            base_dist[i] = float(np.linalg.norm(np.array([x_i, y_i, z_i]) - base))

        # Calculate team progress (only positive values)
        team_progress = 0.0
        for i in self.agents:
            prev = self._prev_target_dist.get(i, curr_dist[i])
            progress = prev - curr_dist[i]  # Positive = getting closer
            team_progress += max(0, progress)  # Only count positive progress

        for i in self.agents:
            if terminations[i] != 1:
                reward[i] = 0

                # Get agent position - CONVERT Vector3r to floats to avoid type errors
                st = self.drone.getMultirotorState(i)
                x, y, z = (
                    float(st.kinematics_estimated.position.x_val),
                    float(st.kinematics_estimated.position.y_val),
                    float(st.kinematics_estimated.position.z_val),
                )
                #y += int(i[-1])*2 # AirSim BUG: spawn offset must be considered! 
                coord[i] = (x,y,z)

                # 🍭 Progress toward goal center (far) vs micro-formation (near)
                target_dist_curr = curr_dist[i]
                base_dist_curr = base_dist[i]
                gate = 1.0 if base_dist_curr <= float(self.form_gate_enable_m) else 0.0

                # Progress to base (task center) when far
                prev_base = self._prev_goal_dist_base.get(i, base_dist_curr)
                prog_base = prev_base - base_dist_curr
                # Progress to micro-formation desired when near
                prev_desired = self._prev_target_dist.get(i, target_dist_curr)
                prog_desired = prev_desired - target_dist_curr

                reward[i] += (1.0 - gate) * float(self.progress_w_far) * prog_base
                reward[i] += gate * float(self.progress_w_near) * prog_desired

                # Remove small proximity bonuses to avoid double counting with progress and success rewards
                
                # Remove extra team-progress bonus to prevent over-incentivizing clustering

                # Small bid entropy bonus to discourage saturated bids
                try:
                    bids = self._last_bids.get(i, np.zeros((self.num_tasks,), dtype=np.float32))
                    p = bids / max(1e-6, float(np.sum(bids))) if float(np.sum(bids)) > 0 else None
                    if p is not None:
                        # entropy in nats; clip to avoid log(0)
                        p_clip = np.clip(p, 1e-6, 1.0)
                        ent = -float(np.sum(p_clip * np.log(p_clip)))
                        reward[i] += float(self.bid_entropy_coef) * ent
                except Exception:
                    pass

                # Penalize moving away from goal (progress < 0), mild
                if prog_base < 0:
                    reward[i] -= (1.0 - gate) * self.away_penalty_far_k * abs(prog_base)
                if prog_desired < 0:
                    reward[i] -= gate * self.away_penalty_near_k * abs(prog_desired)
                self._away_steps[i] = 0

                # 🧱 GENTLE WALL LEARNING: Small penalties that teach without overwhelming
                edge = min(
                    x - self.x_min,
                    self.x_max - x,
                    y - self.y_min,
                    self.y_max - y,
                )
                soft = self.soft_wall_radius
                if edge < soft:
                    frac = (soft - edge) / soft
                    # Wall proximity penalty
                    reward[i] -= 0.1 * (soft - edge)
                    
                    # GENTLE speed penalty near walls (teaches "slow down near walls")
                    if i in act and len(act[i]) >= 2:
                        horiz_mag = float(np.linalg.norm(np.array(act[i][0:2], dtype=np.float32)))
                        reward[i] -= 0.01 * frac * horiz_mag  # Very small penalty

                # Extra penalty on hard geofence hit (after clamping) - REDUCED
                if self._hit_geofence.get(i, False):
                    reward[i] -= 10.0  # Reduced from 50.0 to 10.0 (gentle learning)
                    self._hit_geofence[i] = False

                # 🕐 GENTLE IDLE LEARNING: Very small penalty for extended idling
                try:
                    # Dynamic far threshold based on initial distance at reset (50%)
                    far_thresh = 0.5 * max(1e-6, self._initial_target_dist.get(i, target_dist_curr))
                    far = target_dist_curr > far_thresh
                    # Fix: act is a dict with agent names as keys
                    if i in act:
                        forward_cmd = float(act[i][1]) if len(act[i]) >= 2 else 0.0
                        lateral_cmd = float(act[i][0]) if len(act[i]) >= 1 else 0.0
                        yaw_cmd = float(act[i][3]) if len(act[i]) >= 4 else 0.0
                        translating = (abs(forward_cmd) > 0.05) or (abs(lateral_cmd) > 0.05) or (
                            np.linalg.norm(np.array(act[i][0:2], dtype=np.float32)) > 0.15
                        )
                        yaw_spinning = abs(yaw_cmd) > 0.5
                        if far and (not translating):
                            self._idle_steps[i] = self._idle_steps.get(i, 0) + 1
                            # VERY GENTLE idle penalty (only after extended idling)
                            if self._idle_steps[i] >= 25:  # Increased threshold even more
                                reward[i] -= 0.01  # Tiny penalty (reduced from 0.05)
                        else:
                            self._idle_steps[i] = 0
                except Exception:
                    pass

                # 🚧 GENTLE OBSTACLE LEARNING: Small penalties that teach obstacle avoidance
                try:
                    # Reuse cached depth to avoid extra RPC
                    d_img = self._depth_cache.get(i)
                    if d_img is None:
                        d_img = self.get_depth_image(thresh=2.0, name=i)
                    h, w = d_img.shape[:2]
                    # Center ROI ahead
                    roi = d_img[int(h*0.4):int(h*0.7), int(w*0.4):int(w*0.6)]
                    if roi.size > 0:
                        d_min = float(np.min(roi))
                        safe = 1.5
                        if d_min < safe:
                            # Obstacle proximity penalty (kept small; TTC-any handles action aggressiveness)
                            reward[i] -= 0.1 * (safe - d_min)
                except Exception:
                    pass

                # 🚦 TTC-based any-direction penalty to discourage aggressive approach
                try:
                    heading_range = float(self._last_heading_range.get(i, 1e6))
                    # Use commanded planar speed magnitude as aggressiveness proxy
                    if i in act and len(act[i]) >= 2:
                        cmd_vx = float(act[i][1])
                        cmd_vy = float(act[i][0])
                        cmd_speed = float(np.hypot(cmd_vx, cmd_vy))
                    else:
                        cmd_speed = 0.0
                    if heading_range < 2.5 and cmd_speed > 0.05:  # within 2.5m along current motion
                        # penalty grows with closeness and commanded speed
                        reward[i] -= 0.08 * cmd_speed * (2.5 - heading_range)
                except Exception:
                    pass

                # Removed near-goal vertical deviation penalty to avoid redundancy with ceiling

                # 🎯 GENTLE SMOOTHNESS LEARNING: Very small penalties for jerky movements
                # Fix: act is a dict with agent names as keys
                if i in act and len(act[i]) >= 3:
                    reward[i] -= 0.005 * abs(float(act[i][2]))  # Reduced from 0.02 to 0.005
                if i in act and len(act[i]) >= 4:
                    reward[i] -= 0.005 * abs(float(act[i][3]))  # Reduced from 0.02 to 0.005

                # (Removed per-agent time penalty; replaced with global time-shaped penalty below)

                # Heading/velocity alignment toward goal (applies everywhere; stronger when near)
                try:
                    prev_pos = self._prev_pos.get(i, np.array([x, y, z], dtype=np.float32))
                    vel = (np.array([x, y, z], dtype=np.float32) - prev_pos) / max(self.control_dt, 1e-6)
                    to_goal = self._desired_pos[i] - np.array([x, y, z], dtype=np.float32)
                    dist = np.linalg.norm(to_goal)
                    v_norm = np.linalg.norm(vel)
                    if v_norm > 1e-3 and dist > 1e-3:
                        align = float(np.dot(vel, to_goal) / (v_norm * dist))
                        # weight slightly higher near goal
                        gate = 1.0 if dist < self.near_goal_m else 0.5
                        reward[i] += gate * self.align_k * max(0.0, align)
                except Exception:
                    pass
                    
                # Collision penalty (heavier)
                if self.is_collision(i):
                    reward[i] = -300
                    terminations[i] = 1
                    self.truncations[i] = 1
                    self.obj[i] = -1
                    print(f"Termination reason for {i}: collision")

                # 🚫 High altitude penalty (stay low; learning is near obstacles)
                try:
                    if float(z) < float(self.ceiling_z):
                        over = float(self.ceiling_z) - float(z)
                        reward[i] -= self.ceiling_penalty_per_meter * over
                except Exception:
                    pass

                # 🎉 SUCCESS requires LANDING within target XY perimeter (if enabled)
                if self.require_landing_for_success:
                    # Distance in XY to goal center
                    goal_xy = self._desired_pos[i][:2]
                    dist_xy = float(np.linalg.norm(np.array([x, y], dtype=np.float32) - goal_xy))
                    landed = False
                    try:
                        st_now = self.drone.getMultirotorState(i)
                        landed = (st_now.landed_state == airsim.LandedState.Landed)
                    except Exception:
                        pass
                    if dist_xy < self.landing_radius_m and landed:
                        self._success_hold[i] = self._success_hold.get(i, 0) + 1
                        if self._success_hold[i] >= int(self.success_dwell_steps):
                            # Award success per agent-task; fixed reward only; do not terminate agent
                            try:
                                t_i = int(self._assigned_task.get(i, -1))
                                if 0 <= t_i < len(self.task_value) and t_i not in self._agent_completed_tasks.get(i, set()):
                                    reward[i] += float(self.success_base_reward)
                                    self._agent_completed_tasks[i].add(t_i)
                            except Exception:
                                pass
                            self.obj[i] = 1
                            # Mark agent's task as successful once per agent and update time of last completion
                            try:
                                t_i = int(self._assigned_task.get(i, -1))
                                if 0 <= t_i < len(self._task_success_count):
                                    self._task_success_count[t_i] += 1
                                    # Mark task completed if reached required coalition size
                                    if int(self._task_success_count[t_i]) >= int(self.task_required[t_i]):
                                        self._task_completed[t_i] = True
                                    # Update last completion step
                                    self._last_completion_step = int(self.current_step or 0)
                            except Exception:
                                pass
                            print(f"🎉 {i} landed inside perimeter! +1000")
                    else:
                        # Reset hold if condition breaks
                        self._success_hold[i] = 0
                else:
                    # Legacy proximity-only success (kept as fallback)
                    if target_dist_curr < self.success_radius_m:
                        try:
                            t_i = int(self._assigned_task.get(i, -1))
                            if 0 <= t_i < len(self.task_value) and t_i not in self._agent_completed_tasks.get(i, set()):
                                reward[i] += float(self.success_base_reward)
                                self._agent_completed_tasks[i].add(t_i)
                        except Exception:
                            pass
                        # Removed speed bonus
                        self.obj[i] = 1
                        # Mark agent's task success
                        try:
                            t_i = int(self._assigned_task.get(i, -1))
                            if 0 <= t_i < len(self._task_success_count):
                                self._task_success_count[t_i] += 1
                                if int(self._task_success_count[t_i]) >= int(self.task_required[t_i]):
                                    self._task_completed[t_i] = True
                        except Exception:
                            pass
                        print(f"🎉 {i} got the CANDY! Success: +1000")

                # Update previous position and distance memory at end of per-agent step
                self._prev_target_dist[i] = target_dist_curr
                self._prev_goal_dist_base[i] = base_dist_curr
                self._prev_pos[i] = np.array([x, y, z], dtype=np.float32)

        # Global inter-drone spacing penalty: apply everywhere
        if len(self.agents) > 1:
            for a_id, b_id in combinations(self.agents, 2):
                d_ab = self.msd(coord[a_id], coord[b_id])
                if d_ab < self.safe_spacing_m:
                    penalty = self.spacing_scale * (self.safe_spacing_m - d_ab) * 0.5
                    reward[a_id] -= penalty
                    reward[b_id] -= penalty

        # Pairwise TTC early-warning (only if beyond spacing threshold)
        # Use cached min pairwise TTC per agent from observation step
        try:
            if len(self.agents) > 1:
                # Compute nearest neighbor distance per agent
                nn_dist = {aid: float('inf') for aid in self.agents}
                for a_id, b_id in combinations(self.agents, 2):
                    d_ab = self.msd(coord[a_id], coord[b_id])
                    if d_ab < nn_dist[a_id]:
                        nn_dist[a_id] = d_ab
                    if d_ab < nn_dist[b_id]:
                        nn_dist[b_id] = d_ab
                # Apply TTC penalty only when agent is not already inside spacing radius
                for aid in self.agents:
                    if nn_dist.get(aid, float('inf')) >= float(self.safe_spacing_m):
                        ttc_pair = float(self._last_ttc_pair.get(aid, 1e6))
                        if ttc_pair < float(self.ttc_time_threshold_s):
                            reward[aid] -= float(self.ttc_penalty_scale) * (float(self.ttc_time_threshold_s) - ttc_pair)
        except Exception:
            pass

        # INDIVIDUAL + TEAM REWARDS: Reward individual success, not just team success
        for i in self.agents:
            if self.obj.get(i, 0) == 1:  # Individual success
                reward[i] += 100  # Individual success bonus
                print(f"🎉 {i} individual success bonus: +100")
        
        # Team outcome prints only (avoid extra bonuses/penalties that conflict with task rewards)
        if all([k==1 for k in self.obj.values()]):
            print("################### !!! ALL DRONES ARRIVED !!! ###################")
        elif all([k==-1 for k in self.obj.values()]):
            print("################### ALL DRONES CRASHED :( ###################")

        # Episode success condition: all initially active tasks completed
        try:
            active_init = getattr(self, "_task_active_init", np.ones((self.num_tasks,), dtype=np.float32))
            all_completed = True
            for t in range(self.num_tasks):
                if active_init[t] > 0.5:
                    if t >= len(self._task_completed) or (not bool(self._task_completed[t])):
                        all_completed = False
                        break
            if all_completed:
                for i in self.possible_agents:
                    if self.terminations.get(i, 0) != 1:
                        reward[i] = reward.get(i, 0) + 200.0
                        self.terminations[i] = 1
                        self.truncations[i] = 1
                print("################### ALL ACTIVE TASKS COMPLETED — EPISODE SUCCESS ###################")
        except Exception:
            pass

        # Global time budget and stagnation termination
        try:
            # Remaining time normalized (1 at start → 0 at budget end)
            rt = max(0.0, 1.0 - float(self.current_step or 0) / max(1.0, float(self.global_time_budget_steps)))
            # Time-shaped penalty stronger near deadline
            for i in self.agents:
                if self.terminations.get(i, 0) != 1:
                    reward[i] -= self.time_penalty * 0.5 * (1.5 - rt)  # halved overall
            # Warmup: disable early hard ends for configured episodes
            warmup = (self._episode_index < max(int(self.time_budget_warmup_episodes), int(self.stagnation_warmup_episodes)))
            # Stagnation: end if no completion for too long (unless in warmup)
            if not warmup and int(self.current_step or 0) - int(self._last_completion_step or 0) >= int(self.stagnation_timeout):
                for i in self.possible_agents:
                    if self.terminations.get(i, 0) != 1:
                        self.terminations[i] = 1
                        self.truncations[i] = 1
                print("################### STAGNATION TIMEOUT — EPISODE ENDED ###################")
            # Global time budget: end when exceeded (unless in warmup)
            if not warmup and (self.current_step or 0) >= int(self.global_time_budget_steps):
                for i in self.possible_agents:
                    if self.terminations.get(i, 0) != 1:
                        self.terminations[i] = 1
                        self.truncations[i] = 1
                print("################### GLOBAL TIME BUDGET EXCEEDED — EPISODE ENDED ###################")
        except Exception:
            pass
        # elif all([k==1 for k in self.done.values()]) and any([k==-1 for k in self.obj.values()]) and any([k==1 for k in self.obj.values()]): #try to give a negative reward for each collided proportion to their number and try to give a positive for each arrived
        #     neg = -100*(len([k for k in self.obj.values() if k==-1])/self.num)
        #     pos = 100*(len([k for k in self.obj.values() if k==1])/self.num)
        #     tot = neg + pos
        #     reward = {k:v+tot for k,v in reward.items()}
        #     print("################### SOME ARRIVED, SOME CRUSHED ###################")

        # NO MORE CLIPPING - Let natural reward scale work for proper learning
        # The reward function is now designed to generate reasonable values naturally
        # Clipping was destroying the reward signal differentiation!

        # Debug
        # print("############# Drone n.", i,"#############")
        # print("Agents start pos", x, y, z, " and ", self.agent_start_pos)
        # print("Target pos", self.target_pos)
        # print("Distance origin to target", self.target_dist_prev)
        # print("Traveled x", agent_traveled_x)
        # print("Distance x,y to target", target_dist_curr)
        # print("Rewards", reward)
        # print("#########################################")

        return reward, terminations

    def msd(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    # Multi agent collision
    def is_collision(self, name):
        current_collision_time = self.drone.simGetCollisionInfo(vehicle_name=name).time_stamp
        return True if current_collision_time != self.collision_time[name] else False
    
    # Multi agent rgb view
    def get_rgb_image(self, name):
        rgb_image_request = airsim.ImageRequest(
            0, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request], vehicle_name=name)
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3)) 

        # Sometimes no image returns from api
        try:
            return img2d.reshape(self.image_shape)
        except:
            return np.zeros((self.image_shape))

    # Still to implement multi agent
    def get_depth_image(self, thresh = 2.0, name=None):
        depth_image_request = airsim.ImageRequest(
            1, airsim.ImageType.DepthPerspective, True, False)
        if name is None:
            responses = self.drone.simGetImages([depth_image_request])
        else:
            responses = self.drone.simGetImages([depth_image_request], vehicle_name=name)
        # Some backends may return float64; cast to float32
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = depth_image.reshape(responses[0].height, responses[0].width)
        depth_image[depth_image>thresh]=thresh

        if len(depth_image) == 0:
            depth_image = np.zeros(self.image_shape)

        return depth_image

    def get_lidar_points(self, name):
        try:
            # Try known sensor names
            for s in self._lidar_sensor_names:
                data = self.drone.getLidarData(lidar_name=s, vehicle_name=name)
                if data and len(data.point_cloud) >= 3:
                    pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
                    return pts
            # Fallback: default lidar
            data = self.drone.getLidarData(vehicle_name=name)
            pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
            return pts
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)
