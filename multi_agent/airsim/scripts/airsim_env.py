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
        self.max_steps = 200  # reduced from 250 since drones will move faster now
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
        
        # Control and comm parameters - OPTIMIZED for actual movement
        self.control_dt = 1.7  # reduced from 3.0 to reduce spiral arcs per step
        self.max_xy_speed = 8.0  # increased from 5.0 for faster horizontal motion
        self.max_z_speed = 1.0   # increased from 0.30 for faster vertical motion
        self.max_yaw_rate_deg = 6.0  # lower yaw authority to reduce spirals
        self.smooth_alpha = 0.3  # base action smoothing factor (overridden per-episode)
        # Warmup scheduling for smoothing (reduce at start to encourage visible motion)
        self._episode_index = 0
        self._smooth_warmup_episodes = 2  # reduced from 3
        self._smooth_warmup_value = 0.1  # increased from 0.02
        self._last_action = {i: np.zeros(4, dtype=np.float32) for i in self.possible_agents}
        # Remove extra velocity low-pass; rely on action smoothing only
        self._prev_cmd_v = {i: np.zeros(3, dtype=np.float32) for i in self.possible_agents}

        # Reward shaping and safety tuning - OPTIMIZED for stable learning
        self.gamma = 0.99  # discount used for potential-based shaping
        self.shaping_k = 3.0  # reduced from 5.0 for more stable learning
        self.vertical_shaping_k = 0.3  # reduced from 0.5 for gentler altitude shaping
        self.time_penalty = 0.001  # reduced from 0.007 to prevent penalty accumulation
        self.soft_wall_scale = 0.05  # reduced from 0.2 for much gentler wall penalty
        self.soft_wall_radius = 5.0  # reduced from 8.0 for smaller penalty zone
        self.near_goal_m = 25.0  # increased from 20.0 for wider positive zone
        self.align_k = 1.0  # increased from 0.5 for stronger positive shaping
        self.safe_spacing_m = 1.5  # reduced from 2.0 for tighter formation
        self.spacing_scale = 0.3  # reduced from 0.7 for gentler spacing penalty
        self.success_radius_m = 2.0  # legacy proximity success (kept for shaping)
        # Require landing within target perimeter for final success
        self.require_landing_for_success = True
        self.landing_radius_m = 2.0  # XY radius around target for successful landing
        # Fixed high-altitude discouragement in NED (more negative z = higher)
        self.altitude_threshold_z = -3.0  # penalize when flying above 3 meters
        self.altitude_penalty_per_meter = 5.0  # 5x stronger penalty per meter to discourage high flying
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

        # Geofence (indoor-friendly). XY matches target boundary; Z clamp disabled for debugging
        self.x_min, self.x_max = -150.0, 150.0
        self.y_min, self.y_max = -245.0, 100.0
        # Z clamp enabled - limit height to reasonable indoor flying range
        self.z_min, self.z_max = -5.0, -0.5  # Limit height to 0.5-5 meters

        # Track geofence hits per agent for hard penalties
        self._hit_geofence = {i: False for i in self.possible_agents}

        # Sensor fusion option: include depth into cam channels (keep cam shape HxWx3)
        self.include_depth_in_cam = include_depth_in_cam
        # Exact spawn points only (dict {agent_id: (x,y,z)} or list aligned with agents)
        self.spawn_points = spawn_points
        
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
                        self.drone.moveToZAsync(-1.2, 1.0, vehicle_name=drone_name).join()
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

        obs, info = self.get_obs(self.terminations)
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
            self.drone.moveToZAsync(-0.8, 1.0, vehicle_name=i).join()

        # Set target pose from scene object and derive a start x slightly behind it
        # NOTE: Change object name here if your target differs
        x_t, y_t, z_t = self.drone.simGetObjectPose('Cone4').position
        # Clamp target inside geofence to ensure reachability
        tx = float(np.clip(x_t, self.x_min + 5.0, self.x_max - 5.0))
        ty = float(np.clip(y_t, self.y_min + 5.0, self.y_max - 5.0))
        # Desired Z is at the same height as the cube (AirSim NED: up is negative)
        tz = float(z_t)
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
        self._idle_steps = {}
        self._away_steps = {}
        self._prev_pos = {}
        self._desired_pos = {}
        self._initial_target_dist = {}
        # Disable body-frame long-dt warmup steps; train straight without special debug steps
        self._bf_debug_steps_remaining = {i: 0 for i in self.possible_agents}
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
        # Apply action smoothing
        prev = self._last_action[name]
        act = np.asarray(action, dtype=np.float32)
        act = np.clip(act, -1.0, 1.0)
        # Slightly lower smoothing for quicker response
        smoothed = self.smooth_alpha * prev + (1.0 - self.smooth_alpha) * act
        self._last_action[name] = smoothed
        

        # Map to physical commands (no extra velocity low-pass)
        vx = float(smoothed[1] * self.max_xy_speed)   # forward/back in body frame
        vy = float(smoothed[0] * self.max_xy_speed)   # left/right in body frame
        # AirSim NED: positive z is downward; positive action[2] should mean up -> invert sign
        vz = float(-smoothed[2] * self.max_z_speed)
        # Zero yaw during first few episodes to avoid spirals
        if self._episode_index < 5:
            yaw_rate = 0.0
        else:
            yaw_rate = float(smoothed[3] * self.max_yaw_rate_deg)

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

                # 🍭 CLEAN CANDY-FIRST REWARD: Simple progress toward goal
                target_dist_curr = curr_dist[i]
                
                # SIMPLE PROGRESS: Reward for getting closer to goal (no double-rewarding!)
                prev_dist = self._prev_target_dist.get(i, target_dist_curr)
                progress = prev_dist - target_dist_curr  # Positive = getting closer
                reward[i] += 30.0 * progress  # Strong but not overwhelming
                
                # PROXIMITY BONUS: Extra reward when very close to goal
                if target_dist_curr < 10.0:
                    reward[i] += 15.0  # Significant bonus for being close
                elif target_dist_curr < 25.0:
                    reward[i] += 5.0   # Moderate bonus
                
                # INDIVIDUAL TEAM BONUS: Only reward positive team progress
                team_progress_positive = max(0, team_progress)  # Clip negative values
                reward[i] += 0.5 * (team_progress_positive / max(1, len(self.agents)))

                # Moving-away: disable separate penalty (covered by progress reward)
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
                    # GENTLE wall proximity penalty (much smaller)
                    reward[i] -= 0.1 * (soft - edge)  # Reduced from 0.05 to 0.1, but still small
                    
                    # GENTLE speed penalty near walls (teaches "slow down near walls")
                    if i in act and len(act[i]) >= 2:
                        horiz_mag = float(np.linalg.norm(np.array(act[i][0:2], dtype=np.float32)))
                        reward[i] -= 0.02 * frac * horiz_mag  # Very small penalty

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
                        safe = 1.2
                        if d_min < safe:
                            # GENTLE obstacle penalty (much smaller)
                            reward[i] -= 0.1 * (safe - d_min)  # Reduced from 0.8 to 0.1
                            
                            # GENTLE forward penalty near obstacles (teaches "slow down near obstacles")
                            if i in act and len(act[i]) >= 2 and float(act[i][1]) > 0:
                                reward[i] -= 0.05 * float(act[i][1]) * (safe - d_min)  # Reduced from 0.3 to 0.05
                except Exception:
                    pass

                # 🎯 GENTLE SMOOTHNESS LEARNING: Very small penalties for jerky movements
                # Fix: act is a dict with agent names as keys
                if i in act and len(act[i]) >= 3:
                    reward[i] -= 0.005 * abs(float(act[i][2]))  # Reduced from 0.02 to 0.005
                if i in act and len(act[i]) >= 4:
                    reward[i] -= 0.005 * abs(float(act[i][3]))  # Reduced from 0.02 to 0.005

                # ⏰ GENTLE TIME LEARNING: Very small time pressure (only when far from goal)
                try:
                    far_thresh = 0.5 * max(1e-6, self._initial_target_dist.get(i, target_dist_curr))
                    if target_dist_curr > far_thresh and self.time_penalty > 0.0:
                        reward[i] -= self.time_penalty * 0.1  # Reduce time penalty by 90%
                except Exception:
                    if self.time_penalty > 0.0:
                        reward[i] -= self.time_penalty * 0.1  # Reduce time penalty by 90%

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
                        reward[i] = 1000
                        remaining = max(0, float(self.max_steps - self.current_step))
                        bonus = 0.5 * (remaining / max(1.0, float(self.max_steps))) * 1000.0
                        reward[i] += bonus
                        terminations[i] = 1
                        self.truncations[i] = 1
                        self.obj[i] = 1
                        print(f"🎉 {i} landed inside perimeter! +1000, Speed bonus: +{bonus:.1f}")
                else:
                    # Legacy proximity-only success (kept as fallback)
                    if target_dist_curr < self.success_radius_m:
                        reward[i] = 1000
                        remaining = max(0, float(self.max_steps - self.current_step))
                        bonus = 0.5 * (remaining / max(1.0, float(self.max_steps))) * 1000.0
                        reward[i] += bonus
                        terminations[i] = 1
                        self.truncations[i] = 1
                        self.obj[i] = 1
                        print(f"🎉 {i} got the CANDY! Success: +1000, Speed bonus: +{bonus:.1f}")

                # Add height penalty (encourage staying low)
                if z < self.altitude_threshold_z:  # If flying too high
                    height_penalty = abs(z - self.altitude_threshold_z) * self.altitude_penalty_per_meter
                    reward[i] -= height_penalty
                    if self.debug:
                        print(f"⚠️ {i} flying too high (z={z:.1f}), penalty: -{height_penalty:.1f}")

                # 🎯 PROGRESSIVE LANDING REWARDS: Encourage getting closer to goal progressively
                goal_xy = self._desired_pos[i][:2]
                dist_xy = float(np.linalg.norm(np.array([x, y], dtype=np.float32) - goal_xy))
                
                # Progressive proximity bonus (stronger as you get closer)
                if dist_xy < self.landing_radius_m * 3:  # Within 3x landing radius (6m)
                    proximity_bonus = (self.landing_radius_m * 3 - dist_xy) * 25  # 25 points per meter closer
                    reward[i] += proximity_bonus
                    if self.debug and proximity_bonus > 10:
                        print(f"🎯 {i} near goal (dist={dist_xy:.1f}m), proximity bonus: +{proximity_bonus:.1f}")
                
                # Landing zone bonus (continuous reward while in landing zone)
                if dist_xy < self.landing_radius_m:
                    zone_bonus = 100  # 100 points per step while in landing zone
                    reward[i] += zone_bonus
                    if self.debug:
                        print(f"🛬 {i} in landing zone (dist={dist_xy:.1f}m), zone bonus: +{zone_bonus}")

                # Update previous position and distance memory at end of per-agent step
                self._prev_target_dist[i] = target_dist_curr
                self._prev_pos[i] = np.array([x, y, z], dtype=np.float32)

        # CONDITIONAL inter-drone spacing: Only penalize when far from goal
        if len(self.agents) > 1:
            for a_id, b_id in combinations(self.agents, 2):
                d_ab = self.msd(coord[a_id], coord[b_id])
                if d_ab < self.safe_spacing_m:
                    # Only apply spacing penalty if drones are far from goal (avoid conflicts)
                    a_dist = curr_dist.get(a_id, float('inf'))
                    b_dist = curr_dist.get(b_id, float('inf'))
                    
                    if a_dist > 5.0 and b_dist > 5.0:  # Both drones far from goal
                        penalty = self.spacing_scale * (self.safe_spacing_m - d_ab) * 0.5  # Reduced penalty
                        reward[a_id] -= penalty
                        reward[b_id] -= penalty
                    # If close to goal, ignore spacing penalty to avoid conflicts

        # INDIVIDUAL + TEAM REWARDS: Reward individual success, not just team success
        for i in self.agents:
            if self.obj.get(i, 0) == 1:  # Individual success
                reward[i] += 100  # Individual success bonus
                print(f"🎉 {i} individual success bonus: +100")
        
        # Team bonuses (smaller than individual)
        if all([k==1 for k in self.obj.values()]):
            for i in self.agents:
                reward[i] += 50   # Smaller team bonus
            print("################### !!! ALL DRONES ARRIVED !!! ###################")
        elif all([k==-1 for k in self.obj.values()]):
            for i in self.agents:
                reward[i] -= 50   # Smaller team penalty
            print("################### ALL DRONES CRASHED :( ###################")
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
