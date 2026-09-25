"""Full-frame Siamese CNN + Transformer interception policy.

The Gazebo Actor uses RGB plus measured own velocity and rates; legacy actors
use RGB only. Future target position, collision risk, and
visibility confidence are auxiliary outputs supervised with simulator truth.
The spatial attention map is exported for runtime inspection and debugging.
"""
from typing import List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actions import ACTION_DIM
from .distributions import (
    squashed_normal_log_probability,
    squashed_normal_sample,
)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseResidualBlock(nn.Module):
    """Small MobileNet-style block without a torchvision dependency."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.use_skip = stride == 1 and in_channels == out_channels
        self.depthwise = ConvNormAct(
            in_channels, in_channels, 3, stride, groups=in_channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.pointwise(self.depthwise(inputs))
        if self.use_skip:
            outputs = outputs + inputs
        return self.activation(outputs)


class SpatialAttentionEncoder(nn.Module):
    """Shared frame encoder with an explicit, inspectable attention map."""

    def __init__(self, channels, strides, embedding_dim: int):
        super().__init__()
        channels = [int(value) for value in channels]
        strides = [int(value) for value in strides]
        if len(channels) != len(strides):
            raise ValueError("encoder_channels and encoder_strides must align")
        layers = [ConvNormAct(3, channels[0], 5, 2)]
        previous = channels[0]
        for output, stride in zip(channels, strides):
            layers.append(DepthwiseResidualBlock(previous, output, stride))
            layers.append(DepthwiseResidualBlock(output, output, 1))
            previous = output
        self.features = nn.Sequential(*layers)
        self.attention = nn.Conv2d(previous, 1, kernel_size=1)
        self.projection = nn.Sequential(
            nn.Linear(previous, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, images: torch.Tensor):
        feature_map = self.features(images)
        batch, _, height, width = feature_map.shape
        logits = self.attention(feature_map).flatten(2)
        weights = torch.softmax(logits, dim=-1)
        features = feature_map.flatten(2)
        pooled = (features * weights).sum(dim=-1)
        embedding = self.projection(pooled)
        return embedding, weights.view(batch, height, width)


class ResNet18MultiScaleEncoder(nn.Module):
    """ImageNet ResNet-18 trunk with stride-8/16 feature fusion.

    The stride-8 map preserves the same 80x80 attention resolution used by the
    original encoder at 640x640.  A stride-16 semantic map is projected and
    fused into it, instead of sending thousands of spatial tokens through the
    temporal Transformer.
    """

    def __init__(self, embedding_dim: int, fusion_channels: int = 96,
                 pretrained_weights: str | None = "IMAGENET1K_V1"):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as error:
            raise ImportError(
                "encoder_type=resnet18_multiscale_v1 requires torchvision"
            ) from error

        weight_name = None if pretrained_weights is None else str(pretrained_weights)
        if weight_name is None or weight_name.lower() in ("none", "null", "false"):
            weights = None
        elif weight_name == "DEFAULT":
            weights = ResNet18_Weights.DEFAULT
        elif weight_name == "IMAGENET1K_V1":
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            raise ValueError(
                "pretrained_weights must be null, DEFAULT, or IMAGENET1K_V1"
            )
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        fusion_channels = int(fusion_channels)
        self.lateral_stride8 = nn.Conv2d(128, fusion_channels, 1, bias=False)
        self.lateral_stride16 = nn.Conv2d(256, fusion_channels, 1, bias=False)
        self.fusion = DepthwiseResidualBlock(
            fusion_channels, fusion_channels, stride=1,
        )
        self.attention = nn.Conv2d(fusion_channels, 1, kernel_size=1)
        self.projection = nn.Sequential(
            nn.Linear(fusion_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )

    def pretrained_backbone_parameters(self):
        """Return only parameters initialized from ImageNet."""
        for module in (self.stem, self.layer1, self.layer2, self.layer3):
            yield from module.parameters()

    def set_pretrained_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.pretrained_backbone_parameters():
            parameter.requires_grad_(trainable)

    def keep_pretrained_batch_norm_eval(self) -> None:
        for module in (self.stem, self.layer1, self.layer2, self.layer3):
            for child in module.modules():
                if isinstance(child, nn.BatchNorm2d):
                    child.eval()

    def forward(self, images: torch.Tensor):
        stride4 = self.layer1(self.stem(images))
        stride8 = self.layer2(stride4)
        stride16 = self.layer3(stride8)
        feature_map = self.lateral_stride8(stride8)
        semantic = self.lateral_stride16(stride16)
        semantic = F.interpolate(
            semantic, size=feature_map.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        feature_map = self.fusion(feature_map + semantic)
        batch, _, height, width = feature_map.shape
        logits = self.attention(feature_map).flatten(2)
        weights = torch.softmax(logits, dim=-1)
        pooled = (feature_map.flatten(2) * weights).sum(dim=-1)
        embedding = self.projection(pooled)
        return embedding, weights.view(batch, height, width)


class EndToEndActor(nn.Module):
    """Detector-free image actor and its three auxiliary prediction heads."""

    def __init__(self, cfg_model):
        super().__init__()
        self.history_frames = int(cfg_model.history_frames)
        self.encoder_chunk_size = int(cfg_model.get("encoder_chunk_size", 128))
        if self.history_frames < 1 or self.encoder_chunk_size < 1:
            raise ValueError("history_frames and encoder_chunk_size must be positive")
        embedding_dim = int(cfg_model.embedding_dim)
        encoder_type = str(cfg_model.get("encoder_type", "custom_v1"))
        if encoder_type == "custom_v1":
            self.encoder = SpatialAttentionEncoder(
                cfg_model.encoder_channels,
                cfg_model.get(
                    "encoder_strides",
                    [1] + [2] * (len(cfg_model.encoder_channels) - 1)),
                embedding_dim,
            )
        elif encoder_type == "resnet18_multiscale_v1":
            self.encoder = ResNet18MultiScaleEncoder(
                embedding_dim=embedding_dim,
                fusion_channels=int(cfg_model.get(
                    "encoder_multiscale_channels", 96)),
                pretrained_weights=cfg_model.get(
                    "pretrained_weights", "IMAGENET1K_V1"),
            )
        else:
            raise ValueError(f"unsupported encoder_type: {encoder_type}")
        self.spatial_coordinates = bool(cfg_model.get("spatial_coordinates", False))
        self.visual_yaw_gain = float(cfg_model.get("visual_yaw_gain", 0.0))
        camera_fov = float(cfg_model.get("camera_horizontal_fov", 1.204))
        if not math.isfinite(self.visual_yaw_gain) or self.visual_yaw_gain < 0 or not 0 < camera_fov < math.pi:
            raise ValueError("invalid visual yaw feedback calibration")
        self.camera_tan_half_fov = math.tan(camera_fov/2)
        self.spatial_projection = (nn.Linear(4, embedding_dim, bias=False)
                                   if self.spatial_coordinates else nn.Identity())
        if self.spatial_coordinates:
            nn.init.zeros_(self.spatial_projection.weight)
        self.temporal_position = nn.Parameter(torch.zeros(
            1, self.history_frames, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=int(cfg_model.transformer_heads),
            dim_feedforward=int(cfg_model.transformer_ff_dim),
            dropout=float(cfg_model.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer, num_layers=int(cfg_model.transformer_layers),
            enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.self_state_dim = int(cfg_model.get("self_state_dim", 0))
        if self.self_state_dim not in (0, 6):
            raise ValueError("self_state_dim must be 0 (legacy) or 6 (PX4 own velocity/rates)")
        self.self_state_encoder = (nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 64))
                                   if self.self_state_dim else nn.Identity())
        self.sensor_fusion = (nn.Linear(embedding_dim+64, embedding_dim)
                              if self.self_state_dim else nn.Identity())
        if self.self_state_dim:
            with torch.no_grad():
                self.sensor_fusion.weight.zero_()
                self.sensor_fusion.weight[:, :embedding_dim].copy_(torch.eye(embedding_dim))
                self.sensor_fusion.bias.zero_()
        self.register_buffer("self_state_scale", torch.tensor(
            [float(cfg_model.get("self_velocity_scale", 8.0))]*3
            + [float(cfg_model.get("self_angular_velocity_scale", 2.0))]*3), persistent=False)

        self.action_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embedding_dim, ACTION_DIM),
        )
        self.future_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(embedding_dim // 2, 3),
        )
        self.risk_head = nn.Linear(embedding_dim, 1)
        self.confidence_head = nn.Linear(embedding_dim, 1)
        self.log_std = nn.Parameter(torch.full(
            (ACTION_DIM,), float(cfg_model.init_log_std)))

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        nn.init.normal_(self.temporal_position, std=0.02)
        nn.init.uniform_(self.action_head[-1].weight, -1e-2, 1e-2)
        nn.init.zeros_(self.action_head[-1].bias)

    def _predict(self, frames: torch.Tensor, self_state: Optional[torch.Tensor] = None):
        if frames.dim() != 5:
            raise ValueError("frames must have shape [B,F,3,H,W]")
        batch = frames.size(0)
        history = frames.size(1)
        if history != self.history_frames:
            raise ValueError("unexpected frame history length")
        height, width = frames.size(-2), frames.size(-1)
        images = frames.reshape(batch * history, 3, height, width)
        embeddings, attention = self._encode_images(images)
        tokens = embeddings.view(batch, history, -1)
        tokens = tokens + self.temporal_position
        fused_tokens = self.temporal(tokens)
        fused = self.output_norm(fused_tokens[:, -1])
        if self.self_state_dim:
            if self_state is None:
                raise ValueError("this Actor requires PX4 own velocity and angular velocity")
            if self_state.dim() != 2 or self_state.size(0) != batch or self_state.size(1) != 6:
                raise ValueError("self_state must have shape [B,6]")
            if not bool(torch.isfinite(self_state).all()):
                raise ValueError("self_state contains non-finite sensor values")
            own_embedding = self.self_state_encoder(self_state.float()/self.self_state_scale)
            fused = self.sensor_fusion(torch.cat((fused, own_embedding), dim=-1))

        latent_action = self.action_head(fused)
        future_position = torch.tanh(self.future_head(fused))
        collision_logit = self.risk_head(fused).squeeze(-1)
        confidence_logit = self.confidence_head(fused).squeeze(-1)
        attention = attention.view(
            batch, history, attention.size(-2), attention.size(-1))[:, -1]
        if self.visual_yaw_gain:
            x = (torch.arange(attention.size(-1), device=attention.device, dtype=attention.dtype)+.5)*2/attention.size(-1)-1
            image_x = (attention*x.view(1, 1, -1)).sum(dim=(-2, -1))
            # Camera-calibrated yaw feedback from learned RGB attention;
            # the learned yaw head remains a residual optimized by BC/PPO.
            feedback = self.visual_yaw_gain*torch.atan(image_x*self.camera_tan_half_fov)
            latent_action = torch.cat((latent_action[:, :3], (latent_action[:, 3]+feedback).unsqueeze(-1)), dim=-1)
        return (
            latent_action, future_position, collision_logit,
            confidence_logit, attention,
        )

    def _encode_images(self, images: torch.Tensor):
        """Normalize and encode in chunks to cap BC/PPO activation memory."""
        embeddings = torch.jit.annotate(List[torch.Tensor], [])
        attentions = torch.jit.annotate(List[torch.Tensor], [])
        count = images.size(0)
        for start in range(0, count, self.encoder_chunk_size):
            chunk = images[start:start + self.encoder_chunk_size].float()
            chunk = (chunk / 255.0 - self.image_mean) / self.image_std
            embedding, attention = self.encoder(chunk)
            if self.spatial_coordinates:
                # Retain absolute image position and spread after feature
                # pooling. Coordinates are inferred entirely from RGB.
                h, w = attention.size(-2), attention.size(-1)
                x = (torch.arange(w, device=attention.device, dtype=attention.dtype)+.5)*2/w-1
                y = (torch.arange(h, device=attention.device, dtype=attention.dtype)+.5)*2/h-1
                mx = (attention*x.view(1, 1, w)).sum(dim=(-2, -1))
                my = (attention*y.view(1, h, 1)).sum(dim=(-2, -1))
                xx = (attention*x.square().view(1, 1, w)).sum(dim=(-2, -1))
                yy = (attention*y.square().view(1, h, 1)).sum(dim=(-2, -1))
                embedding = embedding+self.spatial_projection(torch.stack((mx, my, xx, yy), dim=-1))
            embeddings.append(embedding)
            attentions.append(attention)
        return torch.cat(embeddings, dim=0), torch.cat(attentions, dim=0)

    def forward(self, frames: torch.Tensor, self_state: Optional[torch.Tensor] = None):
        """Return action mean, auxiliary predictions, and current attention."""
        latent, future, risk, confidence, attention = self._predict(frames, self_state)
        return torch.tanh(latent), future, risk, confidence, attention

    @torch.no_grad()
    def act(self, frames: torch.Tensor, deterministic: bool = False,
            log_std_min: float = -2.5, log_std_max: float = -0.1,
            self_state: Optional[torch.Tensor] = None):
        latent, future, risk, confidence, attention = self._predict(frames, self_state)
        if deterministic:
            action = torch.tanh(latent)
            log_probability = torch.zeros(
                action.size(0), device=action.device, dtype=action.dtype)
        else:
            log_std = self.log_std.clamp(log_std_min, log_std_max)
            action, log_probability = squashed_normal_sample(latent, log_std)
        return (
            action, log_probability, future, risk, confidence, attention,
        )


class PrivilegedCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, privileged: torch.Tensor) -> torch.Tensor:
        return self.network(privileged).squeeze(-1)


class EndToEndActorCritic(nn.Module):
    """Training container; deployment exports ``actor`` only."""

    def __init__(self, cfg_model):
        super().__init__()
        self.actor = EndToEndActor(cfg_model)
        self.critic = PrivilegedCritic(
            int(cfg_model.critic_dim), int(cfg_model.critic_hidden))

    @torch.no_grad()
    def act(self, frames, privileged, deterministic=False,
            log_std_min=-2.5, log_std_max=-0.1, self_state=None):
        outputs = self.actor.act(
            frames, deterministic, log_std_min, log_std_max, self_state)
        action, log_probability = outputs[0], outputs[1]
        value = self.critic(privileged)
        return (action, log_probability, value, *outputs[2:])

    def evaluate_actions(self, frames, privileged, actions,
                         log_std_min=-2.5, log_std_max=-0.1, self_state=None):
        latent, future, risk, confidence, attention = self.actor._predict(frames, self_state)
        log_std = self.actor.log_std.clamp(log_std_min, log_std_max)
        log_probability = squashed_normal_log_probability(
            actions, latent, log_std)
        _, sampled_log_probability = squashed_normal_sample(latent, log_std)
        entropy = -sampled_log_probability
        value = self.critic(privileged)
        return (
            log_probability, entropy, value,
            future, risk, confidence, attention,
        )
