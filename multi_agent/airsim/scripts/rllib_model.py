from typing import Dict, List, Tuple

import torch as th
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.sample_batch import SampleBatch

# Reuse existing CNN from current codebase
from .network import NatureCNN, GATLayer


class GNNPerAgentModel(TorchModelV2, nn.Module):
    """
    RLlib-compatible per-agent model that preserves per-drone embeddings and
    supports optional team-position attention for coordination.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom = model_config.get("custom_model_config", {})
        if action_space is not None and getattr(action_space, "shape", None) is not None:
            self.action_dim = int(np.prod(action_space.shape))
        else:
            self.action_dim = int(custom.get("action_dim", 3))

        # Cache obs shapes
        try:
            cam_shape = obs_space["cam"].shape  # (H,W,C) likely
            self.cam_hwc = tuple(int(x) for x in cam_shape)
        except Exception:
            self.cam_hwc = (84, 84, 3)
        try:
            pos_shape = obs_space["pos"].shape
            self.pos_dim = int(np.prod(pos_shape))
        except Exception:
            self.pos_dim = 3
        # Optional depth and lidar
        self.use_depth = "depth" in obs_space.spaces if hasattr(obs_space, 'spaces') else False
        self.use_lidar = "lidar" in obs_space.spaces if hasattr(obs_space, 'spaces') else False

        cnn_out_dim = custom.get("cnn_out_dim", 256)
        attn_dim = custom.get("attn_dim", 64)
        gat_hidden = custom.get("gat_hidden", 128)
        gat_heads = custom.get("gat_heads", 2)
        gat_layers = custom.get("gat_layers", 1)

        self.cnn_out_dim = cnn_out_dim
        self.cnn = None
        # Depth encoder (simple 1ch conv) if present
        if self.use_depth:
            self.depth_cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
                nn.Flatten(),
            )
            self.depth_proj = nn.Sequential(nn.Linear(32 * 19 * 19, 128), nn.ReLU())
        # LiDAR MLP if present
        if self.use_lidar:
            self.lidar_mlp = nn.Sequential(
                nn.Linear(2048 * 3, 512), nn.ReLU(),
                nn.Linear(512, 128), nn.ReLU(),
            )

        # Agent index embedding (1-dim to attn_dim)
        self.id_mlp = nn.Sequential(
            nn.Linear(1, attn_dim),
            nn.ReLU(),
        )

        # GAT stack over node features (cam/pos/depth/lidar per drone)
        self.gat_layers = nn.ModuleList([GATLayer(input_dim=cnn_out_dim + self.pos_dim + (128 if self.use_depth else 0) + (128 if self.use_lidar else 0),
                                                  output_dim=gat_hidden, heads=gat_heads) ])
        for _ in range(gat_layers - 1):
            self.gat_layers.append(GATLayer(input_dim=gat_hidden * gat_heads, output_dim=gat_hidden, heads=gat_heads))

        fused_dim = gat_hidden * gat_heads + attn_dim  # GAT node feat + id emb
        self.policy_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * self.action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._last_value: th.Tensor = th.tensor(0.0)

        # Initialize log-std outputs lower to reduce early exploration.
        # The last Linear maps to [mean, log_std]. Bias the log_std half to ~exp(-1.0)=0.37.
        final_lin = None
        for layer in self.policy_head:
            if isinstance(layer, nn.Linear):
                final_lin = layer
        if final_lin is not None and final_lin.bias is not None:
            with th.no_grad():
                # Set lower initial log-std to reduce early exploration (~exp(-1.5)=0.22)
                final_lin.bias[self.action_dim:] = -1.5
                # Make yaw (index 3) even more conservative if present
                if self.action_dim >= 4:
                    final_lin.bias[self.action_dim + 3] = -2.5
                # Slightly reduce coupling into log-std outputs
                final_lin.weight[self.action_dim:, :] *= 0.5

    def _ensure_cnn(self, cam_chw: th.Tensor) -> None:
        if self.cnn is not None:
            return
        c = int(cam_chw.shape[1]) if cam_chw.dim() == 4 else int(cam_chw.shape[0])
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Ensure CNN lives on same device as incoming tensor
        device = cam_chw.device
        self.cnn.to(device)
        with th.no_grad():
            dummy = cam_chw[:1] if cam_chw.dim() == 4 else cam_chw.unsqueeze(0)
            n_flat = self.cnn(dummy.float()).shape[1]
        self.cnn_post = nn.Sequential(
            nn.Linear(n_flat, self.cnn_out_dim),
            nn.ReLU(),
        )
        self.cnn_post.to(device)

    def value_function(self) -> th.Tensor:
        return self._last_value.view(-1)

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        # Ensure we use the model's device for all tensors
        model_device = next(self.parameters()).device

        obs = input_dict.get(SampleBatch.OBS, {})
        if not isinstance(obs, dict):
            obs = {}
        cam = obs.get("cam")
        depth = obs.get("depth") if self.use_depth else None
        lidar = obs.get("lidar") if self.use_lidar else None
        pos = obs.get("pos")
        team_pos = obs.get("team_pos", None)
        agent_idx = obs.get("agent_idx", None)

        B = None

        # Build camera tensor [B,C,H,W] on model device
        if cam is None:
            H, W, C = self.cam_hwc
            B = 1 if pos is None else (1 if (isinstance(pos, th.Tensor) and pos.dim() == 1) or (isinstance(pos, np.ndarray) and pos.ndim == 1) else (pos.shape[0] if hasattr(pos, 'shape') else 1))
            cam_chw = th.zeros((B, C, H, W), dtype=th.float32, device=model_device)
        else:
            if isinstance(cam, np.ndarray):
                cam = th.as_tensor(cam, device=model_device)
            else:
                cam = cam.to(model_device)
            if cam.dim() == 3:  # HWC
                H, W, C = cam.shape
                cam_chw = cam.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
                B = 1
            elif cam.dim() == 4:
                if cam.shape[-1] in (1, 3, 4):
                    B, H, W, C = cam.shape
                    cam_chw = cam.permute(0, 3, 1, 2)
                else:
                    cam_chw = cam
                    B = cam.shape[0]
            else:
                H, W, C = self.cam_hwc
                cam_chw = th.zeros((1, C, H, W), dtype=th.float32, device=model_device)
                B = 1

        # Pos tensor [B,3] on model device
        if pos is None:
            if B is None:
                B = 1
            pos_t = th.zeros((B, 3), dtype=th.float32, device=model_device)
        else:
            if isinstance(pos, np.ndarray):
                pos = th.as_tensor(pos, device=model_device)
            else:
                pos = pos.to(model_device)
            pos_t = pos.unsqueeze(0) if pos.dim() == 1 else pos
            B = pos_t.shape[0]

        # Ensure cam batch matches B
        if cam_chw.shape[0] != B:
            cam_chw = cam_chw.expand(B, -1, -1, -1).contiguous()

        # CNN / depth encoders
        self._ensure_cnn(cam_chw)
        cam_feat = self.cnn_post(self.cnn(cam_chw.float()))  # [B, cnn_out]
        depth_feat = None
        if depth is not None:
            if isinstance(depth, np.ndarray):
                depth = th.as_tensor(depth, device=model_device)
            else:
                depth = depth.to(model_device)
            if depth.dim() == 3:
                depth = depth.permute(2, 0, 1).unsqueeze(0)  # 1,1,H,W
            elif depth.dim() == 4 and depth.shape[-1] == 1:
                depth = depth.permute(0, 3, 1, 2)
            dflat = self.depth_cnn(depth.float())
            depth_feat = self.depth_proj(dflat)

        # Prepare team positions and per-node features
        # Determine team positions tensor and number of nodes N
        team_pos_t = None
        if team_pos is not None:
            if isinstance(team_pos, np.ndarray):
                team_pos_t = th.as_tensor(team_pos, device=model_device).float()
            else:
                team_pos_t = team_pos.to(model_device).float()
            if team_pos_t.dim() == 2:
                team_pos_t = team_pos_t.unsqueeze(0)  # [1, N, 3]
        N = team_pos_t.shape[1] if team_pos_t is not None else 1

        # Relative positions of all teammates to self as node features
        if team_pos_t is not None:
            rel_pos = team_pos_t - pos_t.unsqueeze(1).expand(-1, N, -1)  # [B, N, 3]
        else:
            rel_pos = pos_t.unsqueeze(1).float()  # [B, 1, 3]

        # Tile self features across nodes to match [B, N, *]
        cam_node = cam_feat.unsqueeze(1).expand(-1, N, -1)  # [B, N, cnn_out]
        depth_node = None
        if depth_feat is not None:
            depth_node = depth_feat.unsqueeze(1).expand(-1, N, -1)  # [B, N, 128]
        lidar_node = None
        if lidar is not None:
            if isinstance(lidar, np.ndarray):
                lidar_t = th.as_tensor(lidar, device=model_device)
            else:
                lidar_t = lidar.to(model_device)
            if lidar_t.dim() == 2:
                lidar_t = lidar_t.unsqueeze(0)  # [B, P, 3]
            lidar_flat = lidar_t.reshape(lidar_t.shape[0], -1)  # [B, P*3]
            lidar_feat = self.lidar_mlp(lidar_flat)  # [B, 128]
            lidar_node = lidar_feat.unsqueeze(1).expand(-1, N, -1)  # [B, N, 128]

        feats_by_node = [cam_node, rel_pos]
        if depth_node is not None:
            feats_by_node.append(depth_node)
        if lidar_node is not None:
            feats_by_node.append(lidar_node)
        node_feats = th.cat(feats_by_node, dim=-1)  # [B, N, F]

        # Build adjacency from team_pos if present else trivial [B,1,1]
        if team_pos_t is not None:
            adj = (th.cdist(team_pos_t, team_pos_t) < 100.0).to(node_feats.dtype)  # [B, N, N]
        else:
            adj = th.ones((node_feats.shape[0], 1, 1), dtype=node_feats.dtype, device=node_feats.device)
        # zero diagonal, then ensure rows with no neighbors fall back to self-edge
        eye = th.eye(adj.shape[1], device=node_feats.device, dtype=node_feats.dtype).unsqueeze(0)
        adj = adj * (1 - eye)
        row_sums = adj.sum(dim=2)  # [B, N]
        if row_sums.dim() == 2:
            need_self = (row_sums == 0).to(adj.dtype)  # [B, N]
            if need_self.any():
                eye_b = eye.expand(adj.shape[0], -1, -1)  # [B, N, N]
                adj = adj + eye_b * need_self.unsqueeze(-1)

        # Run GAT stack
        h = node_feats
        for gat in self.gat_layers:
            h = gat(h, adj)
        # Safety: replace any NaN/Inf from degenerate attention with zeros
        h = th.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        # Fuse with id embedding
        if agent_idx is not None:
            if isinstance(agent_idx, np.ndarray):
                agent_idx = th.as_tensor(agent_idx, device=model_device)
            else:
                agent_idx = agent_idx.to(model_device)
            id_emb = self.id_mlp(agent_idx.view(-1, 1).float())
        else:
            id_emb = self.id_mlp(th.zeros((h.shape[0], 1), dtype=th.float32, device=model_device))
        fused = th.cat([h, id_emb.unsqueeze(1).expand(-1, h.shape[1], -1)], dim=-1)

        # Agent id embedding
        if agent_idx is not None:
            if isinstance(agent_idx, np.ndarray):
                agent_idx = th.as_tensor(agent_idx, device=model_device)
            else:
                agent_idx = agent_idx.to(model_device)
            id_emb = self.id_mlp(agent_idx.view(-1, 1).float())
        else:
            id_emb = self.id_mlp(th.zeros((B, 1), dtype=th.float32, device=model_device))

        self.policy_head.to(model_device)
        self.value_head.to(model_device)
        # Pool per-agent fused features via per-agent heads later; use mean pool here
        pooled = fused.mean(dim=1)
        logits = self.policy_head(pooled)
        # Clamp log-std to a tighter range to avoid excessive exploration (no in-place ops)
        if logits.shape[-1] == 2 * self.action_dim:
            mean = logits[:, : self.action_dim]
            log_std_full = logits[:, self.action_dim :]
            # Relax initial exploration a bit
            log_std_full = th.clamp(log_std_full, min=-2.0, max=-0.5)
            if log_std_full.shape[1] >= 4:
                yaw_col = th.clamp(log_std_full[:, 3:4], min=-2.5, max=-1.0)
                log_std_full = th.cat([log_std_full[:, :3], yaw_col, log_std_full[:, 4:]], dim=1)
            logits = th.cat([mean, log_std_full], dim=-1)
        self._last_value = self.value_head(pooled)
        return logits, state