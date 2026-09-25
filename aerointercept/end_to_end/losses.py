"""Auxiliary image supervision shared by BC and PPO, never deployment inputs."""
import torch


def spatial_attention_loss(attention, xy, mask):
    """Cross entropy to a narrow training-only image-center heatmap."""
    h, w = attention.shape[-2:]
    x = (torch.arange(w, device=attention.device, dtype=attention.dtype)+.5)*2/w-1
    y = (torch.arange(h, device=attention.device, dtype=attention.dtype)+.5)*2/h-1
    squared = (x[None, None, :]-xy[:, 0, None, None]).square()
    squared = squared+(y[None, :, None]-xy[:, 1, None, None]).square()
    target = torch.softmax((-squared/(2*.025**2)).flatten(1), dim=-1)
    per_frame = -(target*attention.flatten(1).clamp_min(1e-12).log()).sum(-1)
    return (per_frame*mask).sum()/mask.sum().clamp_min(1.)
