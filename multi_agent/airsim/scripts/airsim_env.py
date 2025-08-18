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

        # Init
        self.drone = airsim.MultirotorClient(ip=ip_address)

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
        self.max_steps = 350  # moderate episode length
        self.current_step = None

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
                            "pos":gymnasium.spaces.Box(
                                low=-500.0, 
                                high=500.0, 
                                shape=(3,), 
                                dtype=np.float32
                            ),
                            # New: depth camera as separate channel (HxWx1, 0..255)
                            "depth": gymnasium.spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.image_shape[0], self.image_shape[1], 1),
                                dtype=np.uint8,
                            ),
                            # New: legacy team positions (kept for backward compat)
                            "team_pos": gymnasium.spaces.Box(
                                low=-500.0,
                                high=500.0,
                                shape=(self.num, 3),
                                dtype=np.float32,
                            ),
                            # New: communication broadcast packet per peer (x,y,z,collision)
                            "team_comm": gymnasium.spaces.Box(
                                low=-2000.0,
                                high=2000.0,
                                shape=(self.num, 4),
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
                        }
                    )
                    for id in self.possible_agents
                }
            )
        
        # Action space (lr, fb, ud, yaw_rate) normalized to [-1, 1]
        self.action_spaces = gymnasium.spaces.Dict(
                {
                    id:gymnasium.spaces.Box(
                        low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                        high=np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32),
                        shape=(4,), 
                        dtype=np.float32
                    )
                    for id in self.possible_agents
                }
            )
        
        # Control and comm parameters
        self.control_dt = 0.25  # command duration per step
        self.max_xy_speed = 5.0  # increase horizontal motion speed
        self.max_z_speed = 0.30   # modest vertical motion
        self.max_yaw_rate_deg = 15.0  # gentler turning to avoid spin-in-place
        self.smooth_alpha = 0.10  # increase responsiveness per step
        # Warmup scheduling for smoothing (reduce at start to encourage visible motion)
        self._episode_index = 0
        self._smooth_warmup_episodes = 3
        self._smooth_warmup_value = 0.02
        self._last_action = {i: np.zeros(4, dtype=np.float32) for i in self.possible_agents}

        # Reward shaping and safety tuning
        self.gamma = 0.99  # discount used for potential-based shaping
        self.shaping_k = 5.0  # stronger distance-progress shaping
        self.vertical_shaping_k = 0.5  # modest altitude shaping
        self.time_penalty = 0.007  # gentle time pressure when far
        self.soft_wall_scale = 0.2  # much softer wall penalty
        self.soft_wall_radius = 8.0  # smaller wall proximity band
        self.near_goal_m = 20.0  # wider alignment/bonus band
        self.align_k = 0.5  # stronger alignment reward near goal
        self.safe_spacing_m = 2.0  # smaller spacing radius indoors
        self.spacing_scale = 0.7  # gentler spacing penalty
        self.success_radius_m = 1.5  # tighter success radius indoors
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

        # Comm parameters
        self.comm_dim = 4  # [x,y,z,collision]
        # Reduce comm noise for faster learning
        self.comm_noise_std = 0.0
        self.comm_drop_prob = 0.0
        self.comm_delay_prob = 0.0
        self._comm_prev = {i: np.zeros((self.num, self.comm_dim), dtype=np.float32) for i in self.possible_agents}
        self._comm_prev_mask = {i: np.zeros((self.num,), dtype=np.float32) for i in self.possible_agents}
        # LiDAR config
        self.max_lidar_points = 2048
        self._lidar_sensor_names = ["LidarSensor1", "Lidar1", "Lidar", "lidar", "LidarSensor"]

        # Geofence (indoor-friendly). XY matches target boundary; Z in NED (negative up)
        # Smaller movement space for smaller map
        self.x_min, self.x_max = -150.0, 150.0
        self.y_min, self.y_max = -245.0, 100.0
        # Slightly relax z_max to avoid constant clamping when hovering
        self.z_min, self.z_max = -2.0, -0.5  # fly close to floor

        # Track geofence hits per agent for hard penalties
        self._hit_geofence = {i: False for i in self.possible_agents}

        # Sensor fusion option: include depth into cam channels (keep cam shape HxWx3)
        self.include_depth_in_cam = include_depth_in_cam
        # Exact spawn points only (dict {agent_id: (x,y,z)} or list aligned with agents)
        self.spawn_points = spawn_points
        
        # Setup flight and set seed
        self.setup_flight()
        self._seed(42)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

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

        # Execute actions for active agents; default to last smoothed action if missing
        for i in self.agents:
            a_i = action.get(i, self._last_action[i])
            self.do_action(a_i, i)

        obs, info = self.get_obs(self.terminations)
        self.reward, self.terminations = self.compute_reward(self.reward, self.terminations, action)

        # PettingZoo agent list update (avoid '__all__')
        self.agents = [k for k in self.possible_agents if self.terminations.get(k, 0) != 1]

        if self.debug:
        print("##################################")
        print("########### Step debug ###########")
        print("##################################")
        print("Returned rewards", self.reward)
        print("Active agents (not dead/not success):", self.agents)
            print("Terminated?", self.terminations)
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
        # Exact spawn points only
        xs, ys, zs = [], [], []
        if self.spawn_points is None:
            raise ValueError("spawn_points must be provided as dict or list of (x,y,z) for each agent")
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
        res = True
        while res:
            y_center = getattr(self, '_spawn_center_y', 0.0)
            # Bias spawn toward increasing Y to keep inside walls and reduce opposite starts
            y_low = max(self.y_min + 5.0, y_center + 10.0)
            y_high = min(self.y_max - 5.0, y_center + 40.0)
            if y_low >= y_high:
                y_low, y_high = self.y_min + 5.0, self.y_max - 5.0
            y = np.random.uniform(y_low, y_high, self.num)
            z_low = max(self.z_min, -1.2)
            z_high = min(self.z_max, -0.9)
            if z_low >= z_high:
                z_low, z_high = self.z_min, self.z_max
            z = np.random.uniform(z_low, z_high, self.num)
            y_combos = combinations(y, 2)
            y_diff = [a-b for a,b in y_combos]
            if all(ele < -5 or ele > 5 for ele in y_diff):
                res = False
        x_val = float(getattr(self, 'agent_start_pos', 0.0))
        x = np.full(self.num, x_val, dtype=np.float32)
        return x, y, z

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
            self._last_action[i] = np.zeros(4, dtype=np.float32)
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
            
            # Take off to a low hover near ground but away from z_max clamp
            self.drone.moveToZAsync(-1.3, 1.0, vehicle_name=i).join()

        # Set target pose from scene object and derive a start x slightly behind it
        # NOTE: Change object name here if your target differs
        x_t, y_t, z_t = self.drone.simGetObjectPose('Cone4').position
        # Clamp target inside geofence to ensure reachability
        tx = float(np.clip(x_t, self.x_min + 5.0, self.x_max - 5.0))
        ty = float(np.clip(y_t, self.y_min + 5.0, self.y_max - 5.0))
        # Desired Z is 2 meters above the cube (AirSim NED: up is negative)
        tz = float(z_t - 2.0)
        # 2D legacy target (kept for compatibility in logs)
        self.target_pos = np.array([tx, ty])
        # 3D base goal position
        self._goal_base_pos = np.array([tx, ty, tz], dtype=np.float32)
        # Choose start x a bit behind the target to reduce wall banging (used for default spawn)
        self.agent_start_pos = float(np.clip(tx - 20.0, self.x_min + 10.0, self.x_max - 10.0))
        # Keep spawn y near target y for shorter initial path (default spawn)
        self._spawn_center_y = ty
      

        # Generate deterministic or constrained spawn positions
        x_pos, y_pos, z_pos = self.generate_pos()

        print("Starting y positions:", y_pos)

        # Enforce minimum start–target XY distance to avoid instant terminations
        min_start_target_xy = 5.0
        tx, ty = float(self.target_pos[0]), float(self.target_pos[1])
        for i in range(self.num):
            # resample y until far enough in XY from target
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

        for i in range(0,self.num):
            pose = airsim.Pose(airsim.Vector3r(float(x_pos[i]), float(y_pos[i]), float(z_pos[i])))
            self.drone.simSetVehiclePose(pose=pose, ignore_collision=True, vehicle_name=self.possible_agents[i])

        # Removed unused self.target_dist_prev

        if self.input_mode == "multi_rgb":
            self.obs_stack = np.zeros(self.image_shape)

        # Get collision time stamp    
        self.collision_time = {i: self.drone.simGetCollisionInfo(vehicle_name=i).time_stamp for i in self.possible_agents}

        # Initialize per-agent previous distance to target and idle counters
        self._prev_target_dist = {}
        self._idle_steps = {}
        self._away_steps = {}
        self._prev_pos = {}
        self._desired_pos = {}
        self._initial_target_dist = {}
        for i in self.possible_agents:
            try:
                x_i, y_i, z_i = self.drone.simGetVehiclePose(i).position
            except Exception:
                x_i, y_i, z_i = 0.0, 0.0, 0.0
            # Desired 3D position for agent i
            self._desired_pos[i] = (self._goal_base_pos + self.agent_offsets[i]).astype(np.float32)
            # Previous 3D error
            d_init = float(np.linalg.norm(np.array([x_i, y_i, z_i]) - self._desired_pos[i]))
            self._prev_target_dist[i] = d_init
            self._initial_target_dist[i] = d_init
            self._idle_steps[i] = 0
            self._away_steps[i] = 0
            self._prev_pos[i] = np.array([x_i, y_i, z_i], dtype=np.float32)


    def do_action(self, action, name):
        # Exponential smoothing
        prev = self._last_action[name]
        act = np.asarray(action, dtype=np.float32)
        act = np.clip(act, -1.0, 1.0)
        smoothed = self.smooth_alpha * prev + (1.0 - self.smooth_alpha) * act
        self._last_action[name] = smoothed

        # Map to physical commands
        vx = float(smoothed[1] * self.max_xy_speed)   # forward/back in body frame
        vy = float(smoothed[0] * self.max_xy_speed)   # left/right in body frame
        # AirSim NED: positive z is downward; positive action[2] should mean up -> invert sign
        vz = float(-smoothed[2] * self.max_z_speed)
        yaw_rate = float(smoothed[3] * self.max_yaw_rate_deg)

        # Optional latency
        # No artificial latency in fast sim mode

        # Execute velocity and yaw concurrently for control_dt
        try:
            f1 = self.drone.moveByVelocityBodyFrameAsync(vx, vy, vz, duration=self.control_dt, vehicle_name=name)
            f2 = self.drone.rotateByYawRateAsync(yaw_rate, duration=self.control_dt, vehicle_name=name)
            f1.join(); f2.join()
        except Exception:
            # Fallback without yaw control
            self.drone.moveByVelocityBodyFrameAsync(vx, vy, vz, duration=self.control_dt, vehicle_name=name).join()

        # Post-action geofence clamp: keep within bounds, zero velocity on hit
        try:
            px, py, pz = self.drone.simGetVehiclePose(name).position
            clamped_x = float(np.clip(px, self.x_min, self.x_max))
            clamped_y = float(np.clip(py, self.y_min, self.y_max))
            clamped_z = float(np.clip(pz, self.z_min, self.z_max))
            if (abs(clamped_x - px) > 1e-3) or (abs(clamped_y - py) > 1e-3) or (abs(clamped_z - pz) > 1e-3):
                pose = airsim.Pose(airsim.Vector3r(clamped_x, clamped_y, clamped_z))
                self.drone.simSetVehiclePose(pose=pose, ignore_collision=True, vehicle_name=name)
                # Damp smoothed action to avoid pushing against wall/ceiling, but allow recovery
                self._last_action[name] *= 0.3
                self._hit_geofence[name] = True
                # Nudge away from wall by adding a small bias opposite to the clamped axis
                nudge = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                if clamped_x in (self.x_min, self.x_max):
                    nudge[0] = -0.2 if clamped_x == self.x_max else 0.2
                if clamped_y in (self.y_min, self.y_max):
                    nudge[1] = -0.2 if clamped_y == self.y_max else 0.2
                # apply a brief nudge in body frame forward direction to re-enter
                try:
        self.drone.moveByVelocityBodyFrameAsync(
                        float(nudge[1]), float(nudge[0]), 0.0, duration=0.2, vehicle_name=name
                    ).join()
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
                x, y, z = self.drone.simGetVehiclePose(agent_id).position
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
                    pos_low = self.observation_spaces[i]["pos"].low
                    pos_high = self.observation_spaces[i]["pos"].high
                    obs[i]['pos'] = np.clip(pos_vec, pos_low, pos_high)
                    # Separate depth channel
                    d_sep = self._depth_cache.get(i)
                    if d_sep is None:
                        d_sep = self.get_depth_image(thresh=1.5, name=i)
                    if d_sep.shape[:2] != (self.image_shape[0], self.image_shape[1]):
                        d_sep = cv2.resize(d_sep, (self.image_shape[1], self.image_shape[0]))
                    d_sep = np.clip(d_sep, 0.0, 1.5)
                    d_sep_u8 = ((d_sep / 1.5) * 255.0).astype(np.uint8)
                    obs[i]['depth'] = np.expand_dims(d_sep_u8, axis=2)
                    # Provide separate depth channel (HxWx1, uint8 0..255)
                    d_sep = self._depth_cache.get(i)
                    if d_sep is None:
                        d_sep = self.get_depth_image(thresh=1.5, name=i)
                    if d_sep.shape[:2] != (self.image_shape[0], self.image_shape[1]):
                        d_sep = cv2.resize(d_sep, (self.image_shape[1], self.image_shape[0]))
                    d_sep = np.clip(d_sep, 0.0, 1.5)
                    d_sep_u8 = ((d_sep / 1.5) * 255.0).astype(np.uint8)
                    obs[i]['depth'] = np.expand_dims(d_sep_u8, axis=2)
                    # Legacy team_pos (still filled)
                    tp_low = self.observation_spaces[i]["team_pos"].low
                    tp_high = self.observation_spaces[i]["team_pos"].high
                    obs[i]['team_pos'] = np.clip(team_pos_full, tp_low, tp_high)

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
                    # Mask out self entry so attention doesn’t collapse onto self
                    self_idx = self.possible_agents.index(i)
                    if 0 <= self_idx < self.num:
                        mask[self_idx] = 0.0
                    # Save for next time (per receiver)
                    self._comm_prev[i] = packets.copy()
                    self._comm_prev_mask[i] = mask.copy()
                    # Clip and assign
                    tc_low = self.observation_spaces[i]["team_comm"].low
                    tc_high = self.observation_spaces[i]["team_comm"].high
                    obs[i]['team_comm'] = np.clip(packets, tc_low, tc_high)
                    obs[i]['team_comm_mask'] = mask.astype(np.float32)
                    obs[i]['agent_idx'] = np.array([idx_norm], dtype=np.float32)
                else:
                    obs[i]['cam'] = np.zeros((self.image_shape), dtype=np.uint8)
                    obs[i]['pos'] = np.zeros((3,), dtype=np.float32)
                    obs[i]['depth'] = np.zeros((self.image_shape[0], self.image_shape[1], 1), dtype=np.uint8)
                    tp_low = self.observation_spaces[i]["team_pos"].low
                    tp_high = self.observation_spaces[i]["team_pos"].high
                    obs[i]['team_pos'] = np.clip(team_pos_full, tp_low, tp_high)
                    # Comm zeroed on termination
                    obs[i]['team_comm'] = np.zeros((self.num, self.comm_dim), dtype=np.float32)
                    obs[i]['team_comm_mask'] = np.zeros((self.num,), dtype=np.float32)
                    obs[i]['agent_idx'] = np.array([idx_norm], dtype=np.float32)
            elif self.input_mode == "depth":
                depth = self.get_depth_image(thresh=3.4).reshape(self.image_shape)
                depth = ((depth/3.4)*255).astype(int)

        if self.debug:
        for i in obs:
                print("Position of", i, ":", obs[i]["pos"])        
        return obs, local_info
    # Multi agent reward
    def compute_reward(self, reward, terminations, act):
        coord = {}

        if self.current_step >= self.max_steps:
            # Distance-proportional timeout penalty per agent:
            # r_timeout_i = -100 * (current_distance / initial_distance)
            reward = {}
            for i in self.possible_agents:
                try:
                    x_i, y_i, z_i = self.drone.simGetVehiclePose(i).position
                except Exception:
                    x_i, y_i, z_i = 0.0, 0.0, 0.0
                desired = self._desired_pos.get(i, self._goal_base_pos)
                curr = float(np.linalg.norm(np.array([x_i, y_i, z_i]) - desired))
                init = float(max(1e-6, self._initial_target_dist.get(i, curr)))
                reward[i] = -100.0 * (curr / init)
            terminations = {i: 1 for i in self.possible_agents}
            self.truncations = {i: 1 for i in self.possible_agents}     
            self.obj = {i: -1 for i in self.possible_agents}
            return reward, terminations

        # Precompute current 3D distances and potential-based progress for all active agents
        curr_dist = {}
        prog = {}
        for i in self.agents:
            x_i, y_i, z_i = self.drone.simGetVehiclePose(i).position
            desired = self._desired_pos.get(i, self._goal_base_pos)
            d = float(np.linalg.norm(np.array([x_i, y_i, z_i]) - desired))
            curr_dist[i] = d
            prev = self._prev_target_dist.get(i, d)
            # Potential-based shaping: k*(prev - gamma*curr)
            prog[i] = prev - self.gamma * d

        team_progress = sum(prog.values()) if prog else 0.0

        for i in self.agents:
            if terminations[i] != 1:
                reward[i] = 0

                # Get agent position
                x,y,z = self.drone.simGetVehiclePose(i).position
                #y += int(i[-1])*2 # AirSim BUG: spawn offset must be considered! 
                coord[i] = (x,y,z)

                # Progress-based reward (3D, potential-based) and proximity shaping
                target_dist_curr = curr_dist[i]
                progress_pb = prog[i]
                reward[i] += self.shaping_k * progress_pb
                # Extra altitude shaping specifically toward desired Z
                desired = self._desired_pos.get(i, self._goal_base_pos)
                alt_err_prev = abs(self._prev_pos[i][2] - desired[2])
                alt_err_curr = abs(z - desired[2])
                reward[i] += self.vertical_shaping_k * (alt_err_prev - self.gamma * alt_err_curr)
                # Proximity shaping: positive reward when within 20m (3D)
                if target_dist_curr < 50.0:
                    reward[i] += 0.2 * (50.0 - target_dist_curr)
                # Small shared team-progress bonus
                reward[i] += 0.3 * (team_progress / max(1, len(self.agents)))

                # Moving-away: disable separate penalty (covered by potential shaping)
                self._away_steps[i] = 0

                # Soft-wall penalty near geofence to discourage straight exits
                edge = min(
                    x - self.x_min,
                    self.x_max - x,
                    y - self.y_min,
                    self.y_max - y,
                )
                soft = self.soft_wall_radius
                if edge < soft:
                    frac = (soft - edge) / soft
                    # Proximity penalty grows as we approach wall (rebalanced, softer)
                    reward[i] -= self.soft_wall_scale * (soft - edge)
                    # Extra penalty for high horizontal speed near wall
                    if len(act[i]) >= 2:
                        horiz_mag = float(np.linalg.norm(np.array(act[i][0:2], dtype=np.float32)))
                        reward[i] -= 0.1 * frac * horiz_mag

                # Extra penalty on hard geofence hit (after clamping)
                if self._hit_geofence.get(i, False):
                    reward[i] -= 50.0
                    self._hit_geofence[i] = False

                # Idle penalty: discourage staying still or yaw-spinning when far from goal
                try:
                    # Dynamic far threshold based on initial distance at reset (50%)
                    far_thresh = 0.5 * max(1e-6, self._initial_target_dist.get(i, target_dist_curr))
                    far = target_dist_curr > far_thresh
                    forward_cmd = float(act[i][1]) if len(act[i]) >= 2 else 0.0
                    lateral_cmd = float(act[i][0]) if len(act[i]) >= 1 else 0.0
                    yaw_cmd = float(act[i][3]) if len(act[i]) >= 4 else 0.0
                    translating = (abs(forward_cmd) > 0.05) or (abs(lateral_cmd) > 0.05) or (
                        np.linalg.norm(np.array(act[i][0:2], dtype=np.float32)) > 0.15
                    )
                    yaw_spinning = abs(yaw_cmd) > 0.5
                    if far and (not translating):
                        self._idle_steps[i] = self._idle_steps.get(i, 0) + 1
                        # Extra penalty when idling via yaw-only spinning
                        if yaw_spinning:
                            reward[i] -= 0.1
                else:
                        self._idle_steps[i] = 0
                    if self._idle_steps[i] >= 5:
                        reward[i] -= 0.3
                except Exception:
                    pass

                # Obstacle proximity penalty using forward depth (shorter range for small map)
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
                        safe = 1.2
                        if d_min < safe:
                            # Base penalty grows as we approach obstacle
                            reward[i] -= 0.8 * (safe - d_min)
                            # Extra when commanding forward
                            if len(act[i]) >= 2 and float(act[i][1]) > 0:
                                reward[i] -= 0.3 * float(act[i][1]) * (safe - d_min)
                except Exception:
                    pass

                # Smoothness penalties (vertical + yaw) — lighter
                if len(act[i]) >= 3:
                    reward[i] -= 0.1 * abs(float(act[i][2]))
                if len(act[i]) >= 4:
                    reward[i] -= 0.1 * abs(float(act[i][3]))

                # Time penalty per step (only when far from goal)
                try:
                    far_thresh = 0.5 * max(1e-6, self._initial_target_dist.get(i, target_dist_curr))
                    if target_dist_curr > far_thresh and self.time_penalty > 0.0:
                        reward[i] -= self.time_penalty
                except Exception:
                    if self.time_penalty > 0.0:
                        reward[i] -= self.time_penalty

                # Heading/velocity alignment near goal
                try:
                    prev_pos = self._prev_pos.get(i, np.array([x, y, z], dtype=np.float32))
                    vel = (np.array([x, y, z], dtype=np.float32) - prev_pos) / max(self.control_dt, 1e-6)
                    to_goal = self._desired_pos[i] - np.array([x, y, z], dtype=np.float32)
                    dist = np.linalg.norm(to_goal)
                    if dist < self.near_goal_m:
                        v_norm = np.linalg.norm(vel)
                        if v_norm > 1e-3:
                            align = float(np.dot(vel, to_goal) / (v_norm * dist))
                            reward[i] += self.align_k * max(0.0, align)
                except Exception:
                    pass
                    
                # Collision penalty (heavier)
                if self.is_collision(i):
                    reward[i] = -300
                    terminations[i] = 1
                    self.truncations[i] = 1
                    self.obj[i] = -1
                    print(f"Termination reason for {i}: collision")

                # Check success in 3D around desired offset point
                if target_dist_curr < self.success_radius_m:
                    reward[i] = 300
                    terminations[i] = 1
                    self.truncations[i] = 1
                    self.obj[i] = 1
                    print(f"Termination reason for {i}: success (within {self.success_radius_m}m 3D)")

                # Update previous position and distance memory at end of per-agent step
                self._prev_target_dist[i] = target_dist_curr
                self._prev_pos[i] = np.array([x, y, z], dtype=np.float32)

        # Smooth inter-drone spacing shaping
        if len(self.agents) > 1:
            for a_id, b_id in combinations(self.agents, 2):
                d_ab = self.msd(coord[a_id], coord[b_id])
                if d_ab < self.safe_spacing_m:
                    penalty = self.spacing_scale * (self.safe_spacing_m - d_ab)
                    reward[a_id] -= penalty
                    reward[b_id] -= penalty

        # Give another reward if all drones reach objective
        if all([k==1 for k in self.obj.values()]):
            reward = {k:v+200 for k,v in reward.items()}
            print("################### !!! ALL DRONES ARRIVED !!! ###################")
        elif all([k==-1 for k in self.obj.values()]):
            reward = {k:v-200 for k,v in reward.items()}
            print("################### ALL DRONES CRASHED :( ###################")
        # elif all([k==1 for k in self.done.values()]) and any([k==-1 for k in self.obj.values()]) and any([k==1 for k in self.obj.values()]): #try to give a negative reward for each collided proportion to their number and try to give a positive for each arrived
        #     neg = -100*(len([k for k in self.obj.values() if k==-1])/self.num)
        #     pos = 100*(len([k for k in self.obj.values() if k==1])/self.num)
        #     tot = neg + pos
        #     reward = {k:v+tot for k,v in reward.items()}
        #     print("################### SOME ARRIVED, SOME CRUSHED ###################")

        # Clip non-terminal per-step rewards to stabilize scale
        for i in self.agents:
            if terminations.get(i, 0) != 1:
                reward[i] = float(np.clip(reward[i], -5.0, 5.0))

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
