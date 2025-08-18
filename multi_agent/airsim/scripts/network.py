import gymnasium
import torch as th
import torch.nn as nn
from typing import Dict, Optional, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.policies import ActorCriticPolicy

class GATLayer(nn.Module):
    """
    Graph Attention Network Layer with stable attention mechanism
    Supports batched inputs.
    """
    def __init__(self, input_dim, output_dim, heads=1, dropout=0.0, alpha=0.2):
        super(GATLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Linear(input_dim, output_dim * heads, bias=False)
        self.attention = nn.Parameter(th.Tensor(1, heads, output_dim))
        nn.init.xavier_uniform_(self.W.weight, gain=0.1)
        nn.init.uniform_(self.attention, -0.1, 0.1)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        """
        x: Node features [N, F] or [B, N, F]
        adj: Adjacency [N, N] or [B, N, N]
        Returns: [N, heads*D] or [B, N, heads*D]
        """
        batched = x.dim() == 3
        if not batched:
            # [N, F] -> [1, N, F]
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
        # x: [B, N, F], adj: [B, N, N]
        B, N, F = x.shape
        Wh = self.W(x.view(B * N, F)).view(B, N, self.heads, self.output_dim)  # [B, N, H, D]
        # Compute attention scores
        a_i = Wh.unsqueeze(2)             # [B, N, 1, H, D]
        a_j = Wh.unsqueeze(1)             # [B, 1, N, H, D]
        attention_input = a_i + a_j       # [B, N, N, H, D]
        e = self.leakyrelu((attention_input * self.attention).sum(dim=-1))  # [B, N, N, H]
        # Mask with adjacency
        e = e.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))  # [B, N, N, H]
        # Softmax over neighbors j (dim=2)
        e = e / 1.0
        att = th.softmax(e, dim=2)  # [B, N, N, H]
        att = self.dropout_layer(att)
        # Aggregate features: sum_j att_{i,j,h} * Wh_{j,h,:}
        h_prime = th.einsum('bijh,bjhd->bihd', att, Wh)  # [B, N, H, D]
        out = h_prime.reshape(B, N, self.heads * self.output_dim)  # [B, N, H*D]
        if not batched:
            out = out.squeeze(0)  # [N, H*D]
        return out

class BasicCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 256):
        super(BasicCNN, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class NatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 512):
        super(NatureCNN, self).__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.LayerNorm(features_dim),  # Changed from BatchNorm1d to LayerNorm
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Convert to float if input is uint8
        if observations.dtype == th.uint8:
            observations = observations.float() / 255.0
        return self.linear(self.cnn(observations))

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gymnasium.spaces.Dict,
        cnn_output_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


def build_adjacency_matrix(positions, radius=100.0):  # Increased radius for larger map
    """
    positions: torch.Tensor
      - [N, 3] (unbatched) or [B, N, 3] (batched)
    radius: threshold distance for connectivity
    Returns:
      - adj: [N, N] or [B, N, N] (same batching as input)
    """
    if positions.dim() == 2:
        # [N, 3]
        dist = th.cdist(positions, positions)  # [N, N]
        adj = (dist < radius).to(positions.dtype)
        adj.fill_diagonal_(0)
        return adj
    elif positions.dim() == 3:
        # [B, N, 3]
        dist = th.cdist(positions, positions)  # [B, N, N]
        adj = (dist < radius).to(positions.dtype)
        # zero out diagonal per batch
        b, n, _ = adj.shape
        eye = th.eye(n, device=positions.device, dtype=positions.dtype).unsqueeze(0).expand(b, -1, -1)
        adj = adj * (1 - eye)
        return adj
    else:
        raise ValueError(f"positions must be [N,3] or [B,N,3], got {positions.shape}")

class GNNCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space, 
        cnn_out_dim=256, 
        gat_hidden=128, 
        gat_heads=4, 
        gat_layers=2
    ):
        super().__init__(observation_space, features_dim=1)

        self.cnn_extractor = NatureCNN(observation_space["cam"], features_dim=cnn_out_dim)
        self.state_extractor = nn.Flatten()
        
        input_dim = cnn_out_dim + get_flattened_obs_dim(observation_space["pos"])
        
        self.gat_layers = nn.ModuleList()
        for _ in range(gat_layers):
            self.gat_layers.append(GATLayer(input_dim, gat_hidden, heads=gat_heads))
            input_dim = gat_hidden * gat_heads

        self._features_dim = input_dim

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        cam = observations["cam"]  # [N,C,H,W] or [B,N,C,H,W]
        pos = observations["pos"]  # [N,3] or [B,N,3]
        # Normalize shapes to batched: [B,N,...]
        if cam.dim() == 4:
            cam = cam.unsqueeze(0)
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        B, N = cam.shape[0], cam.shape[1]
        # CNN over cams
        cams_flat = cam.view(B * N, *cam.shape[2:])  # [B*N, C,H,W]
        f_cam = self.cnn_extractor(cams_flat)        # [B*N, cnn_out]
        f_cam = f_cam.view(B, N, -1)                # [B, N, cnn_out]
        f_pos = pos.view(B, N, -1)                  # [B, N, 3]
        node_feats = th.cat([f_cam, f_pos], dim=-1) # [B, N, cnn_out+3]

        if th.isnan(node_feats).any():
            node_feats = th.nan_to_num(node_feats, nan=0.0)

        adj = build_adjacency_matrix(pos)  # [B, N, N]

        for gat in self.gat_layers:
            node_feats = gat(node_feats, adj)  # keep [B, N, feat]
            if th.isnan(node_feats).any():
                node_feats = th.nan_to_num(node_feats, nan=0.0)

        return node_feats  # [B, N, feature_dim]

class MultiAgentGNNPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        num_drones=3,
        net_arch=None,
        activation_fn=nn.Tanh,
        ortho_init=True,
        use_sde=False,
        log_std_init=0.0,
        full_std=True,
        use_expln=False,
        squash_output=False,
        features_extractor_class=None,
        features_extractor_kwargs=None,
        share_features_extractor=True,
        normalize_images=True,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=None,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs
        )
        
        self.num_drones = num_drones
        self.gnn_extractor = GNNCombinedExtractor(observation_space)
        
        # Use the full feature dimension for each head since we have separate features per drone
        feature_dim = self.gnn_extractor.features_dim
        
        self.action_heads = nn.ModuleList([
            nn.Linear(feature_dim, action_space.shape[0]) 
            for _ in range(num_drones)
        ])
        
        self.value_heads = nn.ModuleList([
            nn.Linear(feature_dim, 1) 
            for _ in range(num_drones)
        ])

        # Initialize weights with very small values to prevent NaN
        for action_head in self.action_heads:
            nn.init.uniform_(action_head.weight, -0.01, 0.01)  # Very small uniform initialization
            if action_head.bias is not None:
                nn.init.constant_(action_head.bias, 0)
        
        for value_head in self.value_heads:
            nn.init.uniform_(value_head.weight, -0.01, 0.01)  # Very small uniform initialization
            if value_head.bias is not None:
                nn.init.constant_(value_head.bias, 0)

    def forward(self, obs, deterministic: bool = False):
        """SB3 acting path: return actions, values, log_prob with batch support.
        Aggregates per-drone outputs to a single action/value per sample.
        """
        logits_per_drone = self.forward_actor(obs)   # [B, N, A]
        values_per_drone = self.forward_critic(obs)  # [B, N, 1]
        # Aggregate across drones
        logits = logits_per_drone.mean(dim=1)  # [B, A]
        values = values_per_drone.mean(dim=1)  # [B, 1]
        # Build distribution and sample
        distribution = self._get_action_dist_from_latent(logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        # Enforce shape
        action_dim = self.action_space.shape[0]
        batch_size = logits.shape[0] if logits.ndim > 1 else 1
        if batch_size == 1:
            actions = actions.view(action_dim).to(th.float32)        # (A,)
        else:
            actions = actions.view(batch_size, action_dim).to(th.float32)  # (B, A)
        return actions, values, log_prob
    
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> "Distribution":
        """
        Retrieve action distribution given the latent codes.
        """
        # Import here to avoid circular imports
        from stable_baselines3.common.distributions import DiagGaussianDistribution, CategoricalDistribution
        
        mean_actions = latent_pi
        
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")
    
    def forward_actor(self, obs):
        """Forward pass for actor (action) network.
        Returns per-drone logits with batch support: [B, N, action_dim].
        """
        drone_features = self.gnn_extractor(obs)  # [B, N, F]
        if drone_features.dim() == 2:
            drone_features = drone_features.unsqueeze(0)  # [1, N, F]
        B, N, F = drone_features.shape
        action_dim = self.action_space.shape[0]
        logits_list = []
        for i in range(self.num_drones):
            logits_list.append(self.action_heads[i](drone_features[:, i, :]))  # [B, A]
        logits = th.stack(logits_list, dim=1)  # [B, N, A]
        return logits
    
    def forward_critic(self, obs):
        """Forward pass for critic (value) network.
        Returns per-drone values with batch support: [B, N, 1].
        """
        drone_features = self.gnn_extractor(obs)  # [B, N, F]
        if drone_features.dim() == 2:
            drone_features = drone_features.unsqueeze(0)  # [1, N, F]
        values_list = []
        for i in range(self.num_drones):
            values_list.append(self.value_heads[i](drone_features[:, i, :]))  # [B, 1]
        values = th.stack(values_list, dim=1)  # [B, N, 1]
        return values

    def evaluate_actions(self, obs, actions) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """SB3 training path using batch-aware per-drone outputs.
        Aggregates over drones to match SB3's single-action-per-sample API.
        """
        logits_per_drone = self.forward_actor(obs)   # [B, N, A]
        values_per_drone = self.forward_critic(obs)  # [B, N, 1]
        # Aggregate over drones (mean) to get shared policy/value per sample
        logits = logits_per_drone.mean(dim=1)  # [B, A]
        values = values_per_drone.mean(dim=1)  # [B, 1]

        action_dim = self.action_space.shape[0]
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if actions.shape[-1] != action_dim:
            actions = actions[..., :action_dim]
        if actions.shape[0] != logits.shape[0]:
            if actions.shape[0] == 1:
                actions = actions.expand(logits.shape[0], -1)
            elif logits.shape[0] == 1:
                logits = logits.expand(actions.shape[0], -1)
                values = values.expand(actions.shape[0], -1)
            else:
                # fallback: align to min batch
                b = min(actions.shape[0], logits.shape[0])
                actions = actions[:b]
                logits = logits[:b]
                values = values[:b]

        distribution = self._get_action_dist_from_latent(logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs) -> "Distribution":
        """SB3 API: return action distribution for given observations."""
        logits = self.forward_actor(obs)  # [num_drones, action_dim]
        return self._get_action_dist_from_latent(logits)

    def predict_values(self, obs) -> th.Tensor:
        """SB3 API: return values for given observations."""
        return self.forward_critic(obs)
    
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> None:
        """
        Override to ensure log_std has correct shape for action space
        """
        super().make_actor(features_extractor)
        
        # Fix log_std to match action space dimensions
        if hasattr(self, 'log_std'):
            if self.log_std.shape[0] != self.action_space.shape[0]:
                print(f"Fixing log_std shape: {self.log_std.shape} -> {self.action_space.shape[0]}")
                # Reinitialize log_std with correct shape for action space
                self.log_std = nn.Parameter(th.ones(self.action_space.shape[0]) * self.log_std_init)