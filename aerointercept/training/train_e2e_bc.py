"""Train the full-frame image actor with behavior cloning and auxiliary labels."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import DotDict, load_config
from ..end_to_end.data import (
    EpisodeSequenceDataset,
    episode_files,
    load_manifest,
    split_episode_files,
)
from ..end_to_end.policy import EndToEndActorCritic
from ..end_to_end.bc_metrics import ActionDiagnostics
from ..end_to_end.losses import spatial_attention_loss
from ..end_to_end.optimization import (
    adamw_with_backbone_lr,
    keep_backbone_batch_norm_eval,
    pretrained_backbone_parameters,
    set_backbone_trainable,
)


@dataclass
class Losses:
    total: torch.Tensor
    action: torch.Tensor
    future: torch.Tensor
    risk: torch.Tensor
    confidence: torch.Tensor
    spatial: torch.Tensor


def validate_initialization_split(source, manifest, validation_files):
    """Reject known scenario leakage when changing a warm-start split."""
    def scenarios(record, names):
        selected = set(names)
        result = set()
        for index, item in enumerate(record.get("summaries", [])):
            name = f"episode_{int(item.get('episode', index)):06d}.npz"
            scenario = item.get("scenario")
            if name in selected and scenario:
                result.add(json.dumps(scenario, sort_keys=True))
        return result
    trained = scenarios(source.get("dataset_manifest", {}),
                        source.get("dataset_split", {}).get("train", []))
    validation = scenarios(manifest, [path.name for path in validation_files])
    if trained & validation:
        raise ValueError("initialization checkpoint already trained on a validation scenario; collect fresh validation episodes")


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def compute_losses(actor, batch, device, auxiliary_cfg, diagnostics=None) -> Losses:
    frames = batch["frames"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    future_target = batch["future_position"].to(device, non_blocking=True)
    risk_target = batch["collision_risk"].to(device, non_blocking=True)
    confidence_target = batch["confidence"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    batch_size, sequence_length = frames.shape[:2]
    own_state = batch.get("self_state")
    if own_state is not None:
        own_state = own_state.to(device, non_blocking=True)
    if getattr(actor, "memory_steps", 0):
        outputs = actor.forward_sequence(frames, own_state)
    else:
        frames = frames.reshape(batch_size * sequence_length, *frames.shape[2:])
        own_state = None if own_state is None else own_state.reshape(batch_size*sequence_length, -1)
        outputs = actor(frames, own_state)
    action_pred, future_pred, risk_logit, confidence_logit, attention = outputs
    action_pred = action_pred.view(batch_size, sequence_length, -1)
    future_pred = future_pred.view(batch_size, sequence_length, -1)
    risk_logit = risk_logit.view(batch_size, sequence_length)
    confidence_logit = confidence_logit.view(batch_size, sequence_length)

    axis_weights = torch.as_tensor(auxiliary_cfg.get("action_axis_weights", [1., 1., 1., 1.]),
                                   dtype=action_pred.dtype, device=action_pred.device)
    if axis_weights.shape != (4,) or not bool(torch.isfinite(axis_weights).all()) or bool((axis_weights <= 0).any()):
        raise ValueError("action_axis_weights must contain four finite positive weights")
    action_mask = mask * confidence_target if auxiliary_cfg.get("visible_action_only", False) else mask
    action_loss = masked_mean(
        ((action_pred - actions).square()*axis_weights).mean(dim=-1), action_mask)
    if diagnostics is not None:
        distances = batch.get("center_distance_m")
        diagnostics.update(action_pred.detach().cpu().numpy(), actions.detach().cpu().numpy(),
                           action_mask.detach().cpu().numpy(),
                           None if distances is None else distances.numpy())
    visible_mask = mask * confidence_target
    future_loss = masked_mean(
        (future_pred - future_target).square().mean(dim=-1), visible_mask)
    risk_loss = masked_mean(
        F.binary_cross_entropy_with_logits(
            risk_logit, risk_target, reduction="none"), mask)
    confidence_loss = masked_mean(
        F.binary_cross_entropy_with_logits(
            confidence_logit, confidence_target, reduction="none"), mask)
    spatial_loss = action_loss.new_zeros(())
    spatial_coef = float(auxiliary_cfg.get("spatial_coef", 0.0))
    if spatial_coef:
        xy = batch["camera_target_xy"].to(device).reshape(-1, 2)
        # Do not supervise a point in black letterbox padding when only the
        # vehicle's legs/rotors remain visible. Source image is 16:9.
        center_visible = (xy[:, 0].abs() <= 1.) & (xy[:, 1].abs() <= 9./16.)
        spatial_loss = spatial_attention_loss(attention, xy, visible_mask.reshape(-1)*center_visible)
    total = (
        float(auxiliary_cfg.get("action_coef", 1.0)) * action_loss
        + auxiliary_cfg.future_coef * future_loss
        + auxiliary_cfg.risk_coef * risk_loss
        + auxiliary_cfg.confidence_coef * confidence_loss
        + spatial_coef * spatial_loss
    )
    return Losses(total, action_loss, future_loss, risk_loss, confidence_loss, spatial_loss)


def run_epoch(actor, loader, device, auxiliary_cfg, optimizer=None,
              keep_batch_norm_eval=False, diagnostics=None):
    training = optimizer is not None
    actor.train(training)
    if training and keep_batch_norm_eval:
        keep_backbone_batch_norm_eval(_ActorContainer(actor))
    totals = {name: 0.0 for name in Losses.__annotations__}
    counts = {name: 0.0 for name in totals}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            losses = compute_losses(actor, batch, device, auxiliary_cfg, diagnostics)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()
            mask = batch["mask"]
            visible = mask*batch["confidence"]
            spatial_count = 0.
            if float(auxiliary_cfg.get("spatial_coef", 0.)):
                xy = batch["camera_target_xy"]
                spatial_count = float((visible*(xy[..., 0].abs() <= 1.)*(xy[..., 1].abs() <= 9./16.)).sum())
            sizes = {"action": float((visible if auxiliary_cfg.get("visible_action_only", False) else mask).sum()),
                     "future": float(visible.sum()), "risk": float(mask.sum()),
                     "confidence": float(mask.sum()), "spatial": spatial_count}
            for name, count in sizes.items():
                totals[name] += float(getattr(losses, name).detach())*count
                counts[name] += count
    result = {name: value/max(1., counts[name]) for name, value in totals.items()}
    result["total"] = sum(result[name]*float(auxiliary_cfg.get(name+"_coef", 1. if name == "action" else 0.))
                          for name in ("action", "future", "risk", "confidence", "spatial"))
    return result


class _ActorContainer:
    """Minimal adapter for shared backbone optimization helpers."""

    def __init__(self, actor):
        self.actor = actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="artifacts/data/e2e_bc")
    parser.add_argument("--out", default="artifacts/runs/training/e2e_bc/best.pt")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--init-temporal-checkpoint", default=None,
                        help="explicitly add a GRU over fused visual and own-state features")
    parser.add_argument("--init-visual-checkpoint", default=None,
                        help="explicitly migrate an image-only checkpoint to image plus own-state")
    parser.add_argument("--init-spatial-checkpoint", default=None,
                        help="explicitly add image position features to an existing Actor")
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--selection-metric", choices=("total", "action"), default="total")
    parser.add_argument("--split", choices=("auto", "random", "mode", "mode_visibility"), default="auto",
                        help="hold out complete episodes within each recorded motion mode")
    parser.add_argument("--visible-action-only", action="store_true",
                        help="use invisible observations for confidence/risk but not truth-only action labels")
    parser.add_argument("--patience", type=int, default=0,
                        help="stop after this many epochs without validation improvement; zero disables")
    parser.add_argument(
        "--backend", choices=("auto", "legacy", "gazebo"), default="auto",
        help="auto selects Gazebo when the dataset manifest records that backend",
    )
    args = parser.parse_args()
    if args.device == "cuda":
        args.device = "cuda:0"
    if args.patience < 0:
        raise ValueError("--patience must be nonnegative")
    if not 0 < args.action_loss_weight < float("inf"):
        raise ValueError("--action-loss-weight must be finite and positive")
    if sum(bool(p) for p in (args.init_checkpoint, args.init_visual_checkpoint, args.init_spatial_checkpoint, args.init_temporal_checkpoint)) > 1:
        raise ValueError("choose only one checkpoint initialization mode")

    manifest = load_manifest(args.data)
    manifest_backend = manifest.get("backend")
    use_gazebo = args.backend == "gazebo" or (
        args.backend == "auto"
        and manifest_backend == "gazebo_harmonic_px4_sitl"
    )
    if args.split == "auto":
        args.split = "mode" if use_gazebo else "random"
    if use_gazebo:
        from ..gazebo.config import load_gazebo_config
        cfg = load_gazebo_config(args.config)
        if cfg.gazebo.task.get("task_version") == "noncontact_rendezvous_v1":
            recorded_task = manifest.get("collection_config", {}).get("task")
            if recorded_task != dict(cfg.gazebo.task):
                raise ValueError("Gazebo dataset has a different or unrecorded task; collect new noncontact data")
    else:
        cfg = load_config(args.config)
    if args.backend == "gazebo" and manifest_backend not in (
        None, "gazebo_harmonic_px4_sitl",
    ):
        raise ValueError("--backend gazebo cannot train from a different backend")
    bc_cfg = cfg.end_to_end.bc
    self_state_dim = int(cfg.end_to_end.model.get("self_state_dim", 0))
    camera_supervision = bool(cfg.end_to_end.auxiliary.get("spatial_coef", 0.0))
    if camera_supervision and manifest.get("collection_config", {}).get("camera_target_label") != "gazebo_center_projection_letterbox_xy_v1":
        raise ValueError("spatial supervision requires recorded camera projection labels")
    if self_state_dim and manifest.get("self_state_source") != "px4_vehicle_odometry_body_frd_v1":
        raise ValueError("new Actor requires newly collected PX4 own-state data; no truth-based reconstruction")
    cfg["end_to_end"]["auxiliary"]["action_coef"] = args.action_loss_weight
    cfg["end_to_end"]["auxiliary"]["visible_action_only"] = args.visible_action_only
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats()
    if int(manifest["history_frames"]) != int(cfg.end_to_end.model.history_frames):
        raise ValueError("dataset/model history_frames mismatch")
    expected_image_size = (
        int(cfg.end_to_end.render.image_width),
        int(cfg.end_to_end.render.image_height),
    )
    dataset_image_size = (
        int(manifest["image_width"]), int(manifest["image_height"]))
    if dataset_image_size != expected_image_size:
        raise ValueError(
            f"dataset image size {dataset_image_size} does not match config "
            f"{expected_image_size}")

    # Legacy datasets require whole-episode success. New collections record
    # measured valid fragments, preserving unsuccessful outcomes for audit.
    all_episode_files = episode_files(args.data)
    if use_gazebo and cfg.gazebo.task.get("task_version") == "noncontact_rendezvous_v1":
        summaries = manifest.get("summaries", [])
        by_name = {
            f"episode_{int(item.get('episode', index)):06d}.npz": item
            for index, item in enumerate(summaries)
        }
        successful = []
        excluded = []
        fragments = manifest.get("collection_config", {}).get("supervision_protocol") == "measured_safe_fragments_v1"
        for path in all_episode_files:
            item = by_name.get(path.name)
            accepted_fragment = False
            if fragments and item is not None and item.get("outcome") in (
                "hit", "fov_lost", "timeout", "contact", "ground", "invalid", "out_of_bounds",
            ):
                with np.load(path, allow_pickle=False) as shard:
                    if "supervision_valid" not in shard:
                        raise ValueError(f"{path}: missing recorded supervision mask")
                    valid = shard["supervision_valid"]
                    if valid.shape != (item["length"],) or not np.isin(valid, [0, 1]).all():
                        raise ValueError(f"{path}: invalid recorded supervision mask")
                    if int(valid.sum()) != item.get("valid_supervision_frames"):
                        raise ValueError(f"{path}: supervision count differs from collection record")
                    accepted_fragment = bool(valid.any())
            if accepted_fragment or (not fragments and item is not None and item.get("outcome") == "hit" and item.get("contact_count") == 0):
                successful.append(path)
            else:
                excluded.append((path.name, (item or {}).get("outcome", "unknown")))
        if not successful:
            raise ValueError("Gazebo dataset contains no valid noncontact supervision")
        if excluded:
            print(f"excluding {len(excluded)} failed Gazebo demonstrations from BC: "
                  f"{dict((outcome, sum(1 for _, value in excluded if value == outcome)) for _, outcome in excluded)}")
        all_episode_files = successful

    sequence_length = args.sequence_length or bc_cfg.sequence_length
    memory_steps = int(cfg.end_to_end.model.get("temporal_memory_steps", 0))
    if memory_steps and sequence_length != memory_steps:
        raise ValueError("temporal BC sequence_length must equal the deployed memory window")
    batch_size = args.batch_size or bc_cfg.batch_size
    epochs = args.epochs or bc_cfg.epochs
    learning_rate = args.learning_rate or bc_cfg.learning_rate
    strata = None
    coverage = None
    if args.split in ("mode", "mode_visibility"):
        summaries_by_name = {
            f"episode_{int(item.get('episode', index)):06d}.npz": item
            for index, item in enumerate(manifest.get("summaries", []))
        }
        strata = [summaries_by_name.get(path.name, {}).get("mode") for path in all_episode_files]
        if not all(strata):
            raise ValueError("mode split requires recorded motion modes; use --split random for legacy data")
        if args.split == "mode_visibility":
            if manifest.get("collection_config", {}).get("supervision_protocol") != "measured_safe_fragments_v1":
                raise ValueError("visibility split requires recorded valid invisible-frame counts")
            coverage = [summaries_by_name[path.name]["valid_invisible_frames"] > 0 for path in all_episode_files]
    train_files, validation_files = split_episode_files(
        all_episode_files, bc_cfg.val_fraction, args.seed, strata=strata, coverage=coverage)
    train_dataset = EpisodeSequenceDataset(
        train_files, sequence_length, cfg.end_to_end.model.history_frames,
        cache_size=int(bc_cfg.get("cache_episodes", 16)), self_state_dim=self_state_dim,
        camera_supervision=camera_supervision)
    validation_dataset = EpisodeSequenceDataset(
        validation_files, sequence_length, cfg.end_to_end.model.history_frames,
        cache_size=int(bc_cfg.get("cache_episodes", 16)), self_state_dim=self_state_dim,
        camera_supervision=camera_supervision)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=int(bc_cfg.num_workers),
        pin_memory=args.device.startswith("cuda"),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_kwargs)

    construction_config = DotDict(dict(cfg.end_to_end.model))
    if args.init_checkpoint or args.init_visual_checkpoint or args.init_spatial_checkpoint or args.init_temporal_checkpoint:
        construction_config["pretrained_weights"] = None
    model = EndToEndActorCritic(construction_config).to(args.device)
    if args.init_temporal_checkpoint:
        from ..gazebo.checkpoint import load_temporal_initialization, validate_task_checkpoint
        source = torch.load(args.init_temporal_checkpoint, map_location=args.device, weights_only=False)
        validate_initialization_split(source, manifest, validation_files)
        if use_gazebo:
            validate_task_checkpoint(source, cfg)
        load_temporal_initialization(model, source, dict(cfg.end_to_end.model))
    if args.init_checkpoint:
        from ..gazebo.checkpoint import load_model_weights, validate_task_checkpoint
        source = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        validate_initialization_split(source, manifest, validation_files)
        if use_gazebo:
            validate_task_checkpoint(source, cfg)
        load_model_weights(model, source, dict(cfg.end_to_end.model))
    if args.init_visual_checkpoint:
        from ..gazebo.checkpoint import load_visual_initialization, validate_task_checkpoint
        source = torch.load(args.init_visual_checkpoint, map_location=args.device, weights_only=False)
        validate_initialization_split(source, manifest, validation_files)
        validate_task_checkpoint(source, cfg)
        load_visual_initialization(model, source, dict(cfg.end_to_end.model))
    if args.init_spatial_checkpoint:
        from ..gazebo.checkpoint import load_spatial_initialization, validate_task_checkpoint
        source = torch.load(args.init_spatial_checkpoint, map_location=args.device, weights_only=False)
        validate_initialization_split(source, manifest, validation_files)
        validate_task_checkpoint(source, cfg)
        load_spatial_initialization(model, source, dict(cfg.end_to_end.model))
    backbone_learning_rate = bc_cfg.get("backbone_learning_rate")
    optimizer = adamw_with_backbone_lr(
        model, learning_rate,
        None if backbone_learning_rate is None else float(backbone_learning_rate),
        actor_only=True,
    )
    freeze_epochs = int(bc_cfg.get("backbone_freeze_epochs", 0))
    keep_batch_norm_eval = bool(
        bc_cfg.get("keep_backbone_batch_norm_eval", False)
    )
    has_pretrained_backbone = bool(pretrained_backbone_parameters(model))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    best_epoch = 0
    started = time.perf_counter()
    train_metrics = {}
    validation_metrics = {}

    print(
        f"end-to-end BC: {len(train_files)} train episodes / "
        f"{len(validation_files)} validation episodes; "
        f"{len(train_dataset)} train windows; backend="
        f"{'gazebo' if use_gazebo else 'legacy'}; "
        f"pretrained_backbone={has_pretrained_backbone}")
    for epoch in range(1, epochs + 1):
        backbone_trainable = epoch > freeze_epochs
        set_backbone_trainable(model, backbone_trainable)
        train_metrics = run_epoch(
            model.actor, train_loader, args.device,
            cfg.end_to_end.auxiliary, optimizer,
            keep_batch_norm_eval=keep_batch_norm_eval)
        action_cfg = cfg.gazebo.action if use_gazebo else cfg.end_to_end.action
        diagnostics = ActionDiagnostics(action_cfg.velocity_max, action_cfg.yaw_rate_max)
        validation_metrics = run_epoch(
            model.actor, validation_loader, args.device,
            cfg.end_to_end.auxiliary, diagnostics=diagnostics)
        with output.with_suffix(".history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "train": train_metrics,
                                     "validation": validation_metrics,
                                     "validation_actions": diagnostics.report()})+"\n")

        saved = ""
        if validation_metrics[args.selection_metric] < best_validation:
            best_validation = validation_metrics[args.selection_metric]
            best_epoch = epoch
            temporary_checkpoint = output.with_suffix(output.suffix + ".tmp")
            torch.save({
                "phase": 3,
                "checkpoint_schema": 7 if memory_steps else 6 if self_state_dim else 5,
                "backend": (
                    "gazebo_harmonic_px4_sitl" if use_gazebo else "legacy_2d"
                ),
                "training_stage": "behavior_cloning",
                "experiment": "C" if use_gazebo else None,
                "model": model.state_dict(),
                "model_config": dict(cfg.end_to_end.model),
                "render_config": dict(cfg.end_to_end.render),
                "action_config": dict(
                    cfg.gazebo.action if use_gazebo else cfg.end_to_end.action
                ),
                "label_config": dict(cfg.end_to_end.labels),
                "auxiliary_config": dict(cfg.end_to_end.auxiliary),
                "task_config": dict(cfg.gazebo.task) if use_gazebo else None,
                "safety_config": dict(cfg.end_to_end.safety),
                "validation_loss": best_validation,
                "validation_metrics": dict(validation_metrics),
                "validation_actions": diagnostics.report(),
                "epoch": epoch,
                "selection_metric": args.selection_metric,
                "action_loss_weight": args.action_loss_weight,
                "initialization_checkpoint": args.init_checkpoint,
                "temporal_initialization_checkpoint": args.init_temporal_checkpoint,
                "visual_initialization_checkpoint": args.init_visual_checkpoint,
                "spatial_initialization_checkpoint": args.init_spatial_checkpoint,
                "self_state_source": manifest.get("self_state_source"),
                "dataset_manifest": manifest,
                "dataset_split": {
                    "strategy": args.split,
                    "train": [path.name for path in train_files],
                    "validation": [path.name for path in validation_files],
                },
                "random_seed": args.seed,
                "image_size": list(expected_image_size),
                "normalization": "float32/255 then ImageNet mean/std",
                "pretraining": {
                    "encoder_type": str(cfg.end_to_end.model.get(
                        "encoder_type", "custom_v1")),
                    "weights": cfg.end_to_end.model.get("pretrained_weights"),
                    "backbone_freeze_epochs": freeze_epochs,
                    "backbone_learning_rate": backbone_learning_rate,
                },
            }, temporary_checkpoint)
            temporary_checkpoint.replace(output)
            saved = " <- saved"
        print(
            f"epoch {epoch:3d}/{epochs} "
            f"train={train_metrics['total']:.4f} "
            f"val={validation_metrics['total']:.4f} "
            f"action={validation_metrics['action']:.4f} "
            f"future={validation_metrics['future']:.4f} "
            f"risk={validation_metrics['risk']:.4f} "
            f"conf={validation_metrics['confidence']:.4f} "
            f"spatial={validation_metrics['spatial']:.4f} "
            f"backbone={'train' if backbone_trainable else 'frozen'}{saved}")
        if args.patience and epoch-best_epoch >= args.patience and epoch > freeze_epochs:
            print(f"early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "experiment": "C" if use_gazebo else None,
        "backend": "gazebo_harmonic_px4_sitl" if use_gazebo else "legacy_2d",
        "device": args.device,
        "epochs": int(epoch),
        "requested_epochs": int(epochs),
        "best_epoch": best_epoch,
        "patience": args.patience,
        "train_episodes": len(train_files),
        "validation_episodes": len(validation_files),
        "dataset_split": {
            "strategy": args.split,
            "train": [path.name for path in train_files],
            "validation": [path.name for path in validation_files],
        },
        "train_windows": len(train_dataset),
        "best_validation_loss": best_validation,
        "selection_metric": args.selection_metric,
        "action_loss_weight": args.action_loss_weight,
        "auxiliary_config": dict(cfg.end_to_end.auxiliary),
        "initialization_checkpoint": args.init_checkpoint,
        "temporal_initialization_checkpoint": args.init_temporal_checkpoint,
        "visual_initialization_checkpoint": args.init_visual_checkpoint,
        "spatial_initialization_checkpoint": args.init_spatial_checkpoint,
        "self_state_source": manifest.get("self_state_source"),
        "final_train_metrics": train_metrics,
        "final_validation_metrics": validation_metrics,
        "elapsed_seconds": elapsed,
        "gpu_peak_torch_mb": (
            torch.cuda.max_memory_allocated() / 2**20
            if args.device.startswith("cuda") else 0.0
        ),
        "checkpoint": str(output),
    }
    metrics_path = output.with_suffix(output.suffix + ".metrics.json")
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"best validation loss {best_validation:.5f}; saved {output}")
    print("AEROINTERCEPT_BC=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
