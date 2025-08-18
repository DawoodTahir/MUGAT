"""
Minimal env wrapper for MAGNNET-style task exposure.
Keeps interface similar to existing AirSimDroneEnv but adds task_pos/task_mask
in observations. Import and use this env if you want to try MAGNNET quickly
without touching your current env.
"""
from typing import Tuple, Dict
import numpy as np


def add_tasks_to_obs(obs: Dict[str, Dict], tasks_xyz: np.ndarray, task_mask: np.ndarray) -> Dict[str, Dict]:
    """
    Injects 'task_pos' and 'task_mask' into each agent's obs dict.
    tasks_xyz: (K,3) float32, task_mask: (K,) {0,1}
    """
    tasks_xyz = tasks_xyz.astype(np.float32)
    task_mask = task_mask.astype(np.float32)
    out = {}
    for aid, od in obs.items():
        od2 = dict(od)
        od2["task_pos"] = tasks_xyz
        od2["task_mask"] = task_mask
        out[aid] = od2
    return out


def synthesize_ring_tasks(center_xyz: Tuple[float, float, float], radius: float, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple ring of K tasks around center (x,y,z). Returns (task_pos(K,3), task_mask(K,)).
    """
    cx, cy, cz = center_xyz
    angles = np.linspace(0, 2 * np.pi, num=k, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    zs = np.full_like(xs, cz, dtype=np.float32)
    tasks = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    mask = np.ones((k,), dtype=np.float32)
    return tasks, mask

