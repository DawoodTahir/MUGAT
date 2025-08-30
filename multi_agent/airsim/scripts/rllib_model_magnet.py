from typing import Dict, List, Tuple

import torch as th
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.sample_batch import SampleBatch

# Reuse existing CNN/GAT from current codebase
from .network import GATLayer


class GNNPerAgentModelMAGNET(TorchModelV2, nn.Module):
    """
    MAGNNET-style extension: adds an agent↔task attention head and uses
    attended task context to condition the continuous control head.

    Action space remains continuous (same shape as your current model),
    so you can drop this model into PPO without changing the trainer.

    Expected extra obs keys (per agent):
      - task_pos:  (K,3) or (B,K,3) float32
      - task_mask: (K,)  or (B,K)    {0,1}
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
            cam_shape = obs_space["cam"].shape
            self.cam_hwc = tuple(int(x) for x in cam_shape)
        except Exception:
            self.cam_hwc = (84, 84, 3)
        try:
            self.pos_dim = int(np.prod(obs_space["pos"].shape))
        except Exception:
            self.pos_dim = 3

        # Flags/dims
        cnn_out_dim = int(custom.get("cnn_out_dim", 256))
        attn_dim = int(custom.get("attn_dim", 64))
        gat_hidden = int(custom.get("gat_hidden", 128))
        gat_heads = int(custom.get("gat_heads", 2))
        gat_layers = int(custom.get("gat_layers", 1))
        self.cnn_out_dim = cnn_out_dim

        # CNN build-on-first-use
        self.cnn = None
        # Depth/LiDAR not used here for brevity; can be added similar to base model

        # Agent id embedding (1 -> attn_dim)
        self.id_mlp = nn.Sequential(nn.Linear(1, attn_dim), nn.ReLU())

        # Team GAT
        self.gat_layers = nn.ModuleList([
            GATLayer(input_dim=cnn_out_dim + self.pos_dim, output_dim=gat_hidden, heads=gat_heads)
        ])
        for _ in range(gat_layers - 1):
            self.gat_layers.append(GATLayer(input_dim=gat_hidden * gat_heads, output_dim=gat_hidden, heads=gat_heads))

        fused_dim = gat_hidden * gat_heads + attn_dim

        # MAGNNET task attention modules
        self.task_feat_dim = int(custom.get("magnet_task_feat_dim", 3))  # default xyz
        self.task_encoder = nn.Sequential(
            nn.Linear(self.task_feat_dim, 128), nn.ReLU(),
            nn.Linear(128, attn_dim)
        )
        self.agent_query = nn.Linear(fused_dim, attn_dim)
        self.task_context_proj = nn.Linear(attn_dim, 128)
        self.fused_for_policy = nn.Linear(fused_dim + 128, fused_dim)

        # Message-aware Q/K/V for peer comm (unified team_comm)
        self.msg_key = nn.Linear(64, attn_dim)
        self.msg_val = nn.Linear(64, attn_dim)
        self.msg_query = nn.Linear(fused_dim, attn_dim)
        # Project message context to fused_dim so pooled stays same size
        self.msg_ctx_proj2 = nn.Linear(attn_dim, fused_dim)

        # LiDAR fusion (quick path): encode 32-bin LiDAR and fuse into pooled via residual
        self.lidar_mlp = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.lidar_proj2 = nn.Linear(64, fused_dim)

        # TTC fusion (self-safety context)
        self.ttc_mlp = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
        )
        self.ttc_proj = nn.Linear(32, fused_dim)

        # Continuous control head (same format as base): [mean, log_std]
        self.policy_head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(),
            nn.Linear(256, 2 * self.action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Init conservative log_std bias (like base)
        final_lin = None
        for layer in self.policy_head:
            if isinstance(layer, nn.Linear):
                final_lin = layer
        if final_lin is not None and final_lin.bias is not None:
            with th.no_grad():
                final_lin.bias[self.action_dim:] = -1.5
                if self.action_dim >= 4:
                    final_lin.bias[self.action_dim + 3] = -2.5
                final_lin.weight[self.action_dim:, :] *= 0.5

        self._last_value: th.Tensor = th.tensor(0.0)

    def _ensure_cnn(self, cam_chw: th.Tensor) -> None:
        if self.cnn is not None:
            return
        c = int(cam_chw.shape[1]) if cam_chw.dim() == 4 else int(cam_chw.shape[0])
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        device = cam_chw.device
        self.cnn.to(device)
        with th.no_grad():
            dummy = cam_chw[:1] if cam_chw.dim() == 4 else cam_chw.unsqueeze(0)
            n_flat = self.cnn(dummy.float()).shape[1]
        self.cnn_post = nn.Sequential(nn.Linear(n_flat, self.cnn_out_dim), nn.ReLU())
        self.cnn_post.to(device)

    def value_function(self) -> th.Tensor:
        return self._last_value.view(-1)

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        device = next(self.parameters()).device
        obs = input_dict.get(SampleBatch.OBS, {})
        if not isinstance(obs, dict):
            obs = {}

        cam = obs.get("cam")
        pos = obs.get("pos")
        team_pos = obs.get("team_pos", None)
        team_comm = obs.get("team_comm", None)                # (N,8) unified
        team_comm_mask = obs.get("team_comm_mask", None)      # (N,)
        lidar_bins = obs.get("lidar", None)                   # (32,)
        agent_idx = obs.get("agent_idx", None)
        ttc = obs.get("ttc", None)

        # Camera tensor [B,C,H,W]
        if cam is None:
            H, W, C = self.cam_hwc
            if pos is None:
                B = 1
            else:
                if isinstance(pos, th.Tensor) and pos.dim() == 1:
                    B = 1
                elif isinstance(pos, np.ndarray) and np.ndim(pos) == 1:
                    B = 1
                else:
                    B = pos.shape[0]
            cam_chw = th.zeros((B, C, H, W), dtype=th.float32, device=device)
        else:
            if isinstance(cam, np.ndarray):
                cam = th.as_tensor(cam, device=device)
            else:
                cam = cam.to(device)
            if cam.dim() == 3:  # HWC
                H, W, C = cam.shape
                cam_chw = cam.permute(2, 0, 1).unsqueeze(0)
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
                cam_chw = th.zeros((1, C, H, W), dtype=th.float32, device=device)
                B = 1

        # Position [B,3]
        if pos is None:
            pos_t = th.zeros((B, 3), dtype=th.float32, device=device)
        else:
            if isinstance(pos, np.ndarray):
                pos = th.as_tensor(pos, device=device)
            else:
                pos = pos.to(device)
            pos_t = pos.unsqueeze(0) if pos.dim() == 1 else pos
            B = pos_t.shape[0]

        if cam_chw.shape[0] != B:
            cam_chw = cam_chw.expand(B, -1, -1, -1).contiguous()

        # CNN features
        self._ensure_cnn(cam_chw)
        cam_feat = self.cnn_post(self.cnn(cam_chw.float()))  # [B, cnn_out]

        # Team graph features
        team_pos_t = None
        if team_pos is not None:
            if isinstance(team_pos, np.ndarray):
                team_pos_t = th.as_tensor(team_pos, device=device).float()
            else:
                team_pos_t = team_pos.to(device).float()
            if team_pos_t.dim() == 2:
                team_pos_t = team_pos_t.unsqueeze(0)
        N = team_pos_t.shape[1] if team_pos_t is not None else 1

        if team_pos_t is not None:
            rel_pos = team_pos_t - pos_t.unsqueeze(1).expand(-1, N, -1)
        else:
            rel_pos = pos_t.unsqueeze(1).float()

        cam_node = cam_feat.unsqueeze(1).expand(-1, N, -1)

        # Message-aware features: encode team_comm per peer (for Q/K/V only)
        msg_feat = None
        if team_comm is not None:
            if isinstance(team_comm, np.ndarray):
                team_comm_t = th.as_tensor(team_comm, device=device).float()
            else:
                team_comm_t = team_comm.to(device).float()
            if team_comm_t.dim() == 2:
                team_comm_t = team_comm_t.unsqueeze(0)
            msg_in = team_comm_t  # (B,N,8)
            if not hasattr(self, "_msg_mlp"):
                in_dim = msg_in.shape[-1]
                self._msg_mlp = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU())
                self._msg_mlp.to(device)
            msg_feat = self._msg_mlp(msg_in)

        node_feats = th.cat([cam_node, rel_pos], dim=-1)

        if team_pos_t is not None:
            adj = (th.cdist(team_pos_t, team_pos_t) < 100.0).to(node_feats.dtype)
        else:
            adj = th.ones((node_feats.shape[0], 1, 1), dtype=node_feats.dtype, device=node_feats.device)
        eye = th.eye(adj.shape[1], device=node_feats.device, dtype=node_feats.dtype).unsqueeze(0)
        adj = adj * (1 - eye)
        # Optional hard validity gating: silence invalid senders before attention
        if team_comm_mask is not None:
            if isinstance(team_comm_mask, np.ndarray):
                mask_t = th.as_tensor(team_comm_mask, device=device).float()
            else:
                mask_t = team_comm_mask.to(device).float()
            if mask_t.dim() == 1:
                mask_t = mask_t.unsqueeze(0)
            # shape [B,1,N] broadcast over receivers
            sender_mask = mask_t.unsqueeze(1)
            adj = adj * sender_mask
        row_sums = adj.sum(dim=2)
        if row_sums.dim() == 2:
            need_self = (row_sums == 0).to(adj.dtype)
            if need_self.any():
                eye_b = eye.expand(adj.shape[0], -1, -1)
                adj = adj + eye_b * need_self.unsqueeze(-1)

        h = node_feats
        for gat in self.gat_layers:
            h = gat(h, adj)
        h = th.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        # Agent id embedding
        if agent_idx is not None:
            if isinstance(agent_idx, np.ndarray):
                agent_idx = th.as_tensor(agent_idx, device=device)
            else:
                agent_idx = agent_idx.to(device)
            id_emb = self.id_mlp(agent_idx.view(-1, 1).float())
        else:
            id_emb = self.id_mlp(th.zeros((h.shape[0], 1), dtype=th.float32, device=device))

        fused = th.cat([h, id_emb.unsqueeze(1).expand(-1, h.shape[1], -1)], dim=-1)
        pooled = fused.mean(dim=1)  # per-agent embedding

        # Per-peer message attention (Q/K/V) over local neighbors using message embeddings
        if msg_feat is not None:
            q_peer = self.msg_query(pooled).unsqueeze(1)        # (B,1,attn)
            k_peer = self.msg_key(msg_feat)                     # (B,N,attn)
            v_peer = self.msg_val(msg_feat)                     # (B,N,attn)
            # Mask invalid senders
            if team_comm_mask is not None:
                if isinstance(team_comm_mask, np.ndarray):
                    m = th.as_tensor(team_comm_mask, device=device).float()
                else:
                    m = team_comm_mask.to(device).float()
                if m.dim() == 1:
                    m = m.unsqueeze(0)
                very_neg = th.finfo(q_peer.dtype).min / 2
                logits = (q_peer * k_peer).sum(dim=-1)  # (B,N)
                logits = logits.masked_fill(m < 0.5, very_neg)
            else:
                logits = (q_peer * k_peer).sum(dim=-1)
            attn = th.softmax(logits, dim=-1).unsqueeze(-1)     # (B,N,1)
            msg_ctx = (attn * v_peer).sum(dim=1)                # (B,attn)
            # Fuse message context into pooled features (project to fused_dim)
            pooled = pooled + self.msg_ctx_proj2(msg_ctx)
        # LiDAR quick fusion (residual)
        if lidar_bins is not None:
            if isinstance(lidar_bins, np.ndarray):
                lidar_t = th.as_tensor(lidar_bins, device=device).float()
            else:
                lidar_t = lidar_bins.to(device).float()
            if lidar_t.dim() == 1:
                lidar_t = lidar_t.unsqueeze(0)
            lidar_feat = self.lidar_mlp(lidar_t)
            # Reduce residual weight further and gate by a sigmoid to avoid dominance
            lidar_ctx = th.sigmoid(self.lidar_proj2(lidar_feat))
            pooled = pooled + 0.1 * lidar_ctx

        # MAGNNET task attention
        task_pos = obs.get("task_pos", None)
        task_mask = obs.get("task_mask", None)
        fused_for_policy = pooled
        if task_pos is not None:
            if isinstance(task_pos, np.ndarray):
                task_pos_t = th.as_tensor(task_pos, device=device).float()
            else:
                task_pos_t = task_pos.to(device).float()
            # Ensure (B,K,feat)
            if task_pos_t.dim() == 2:
                task_pos_t = task_pos_t.unsqueeze(0)
            B2, K, feat = task_pos_t.shape
            if task_mask is not None:
                if isinstance(task_mask, np.ndarray):
                    task_mask_t = th.as_tensor(task_mask, device=device).float()
                else:
                    task_mask_t = task_mask.to(device).float()
                if task_mask_t.dim() == 1:
                    task_mask_t = task_mask_t.unsqueeze(0)
            else:
                task_mask_t = th.ones((B2, K), dtype=th.float32, device=device)

            # Encode tasks and compute attention
            task_k = self.task_encoder(task_pos_t.reshape(B2 * K, feat)).reshape(B2, K, -1)
            q = self.agent_query(pooled)           # (B, attn_dim)
            q = q.unsqueeze(1)                     # (B, 1, attn_dim)
            scores = (q * task_k).sum(dim=-1)      # (B, K)
            very_neg = th.finfo(scores.dtype).min / 2
            scores = scores.masked_fill(task_mask_t < 0.5, very_neg)
            attn = th.softmax(scores, dim=1)       # (B, K)
            ctx = (attn.unsqueeze(-1) * task_k).sum(dim=1)  # (B, attn_dim)
            ctx_proj = self.task_context_proj(ctx)          # (B, 128)
            fused_for_policy = th.relu(self.fused_for_policy(th.cat([pooled, ctx_proj], dim=-1)))

        # Fuse TTC scalars into pooled (self-only safety context)
        if ttc is not None:
            if isinstance(ttc, np.ndarray):
                ttc_t = th.as_tensor(ttc, device=device).float()
            else:
                ttc_t = ttc.to(device).float()
            if ttc_t.dim() == 1:
                ttc_t = ttc_t.unsqueeze(0)
            # Simple bounded features then project
            ttc_feat = th.clamp(ttc_t, 0.0, 100.0)
            pooled = pooled + self.ttc_proj(self.ttc_mlp(ttc_feat))

        # Heads
        logits = self.policy_head(fused_for_policy)
        if logits.shape[-1] == 2 * self.action_dim:
            mean = logits[:, : self.action_dim]
            log_std_full = logits[:, self.action_dim:]
            log_std_full = th.clamp(log_std_full, min=-2.0, max=-0.5)
            if log_std_full.shape[1] >= 4:
                yaw_col = th.clamp(log_std_full[:, 3:4], min=-2.5, max=-1.0)
                log_std_full = th.cat([log_std_full[:, :3], yaw_col, log_std_full[:, 4:]], dim=1)
            logits = th.cat([mean, log_std_full], dim=-1)

        self._last_value = self.value_head(fused_for_policy)
        return logits, state

