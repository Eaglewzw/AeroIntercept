"""Optimizer grouping for pretrained visual backbones."""

from __future__ import annotations

import torch


def pretrained_backbone_parameters(model) -> list[torch.nn.Parameter]:
    encoder = model.actor.encoder
    getter = getattr(encoder, "pretrained_backbone_parameters", None)
    return list(getter()) if getter is not None else []


def adamw_with_backbone_lr(
    model,
    learning_rate: float,
    backbone_learning_rate: float | None = None,
    weight_decay: float = 1.0e-4,
    actor_only: bool = False,
) -> torch.optim.AdamW:
    """Use a conservative LR for ImageNet weights and the normal LR elsewhere."""
    backbone = pretrained_backbone_parameters(model)
    optimized_module = model.actor if actor_only else model
    if not backbone or backbone_learning_rate is None:
        return torch.optim.AdamW(
            optimized_module.parameters(), lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    backbone_ids = {id(parameter) for parameter in backbone}
    task_parameters = [
        parameter for parameter in optimized_module.parameters()
        if id(parameter) not in backbone_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(backbone_learning_rate)},
            {"params": task_parameters, "lr": float(learning_rate)},
        ],
        weight_decay=float(weight_decay),
    )


def set_backbone_trainable(model, trainable: bool) -> None:
    setter = getattr(
        model.actor.encoder, "set_pretrained_backbone_trainable", None,
    )
    if setter is not None:
        setter(bool(trainable))


def keep_backbone_batch_norm_eval(model) -> None:
    setter = getattr(
        model.actor.encoder, "keep_pretrained_batch_norm_eval", None,
    )
    if setter is not None:
        setter()


def keep_actor_batch_norm_eval(actor) -> None:
    """Keep deployed normalization during small corrective-data fine-tuning.

    Sequential image chunks contain strongly correlated frames. Updating their
    running statistics on a small corrective dataset can change the deployed
    policy even when the corresponding weights are frozen. Affine parameters
    remain trainable; dropout and other training modules are unaffected.
    """
    for module in actor.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
