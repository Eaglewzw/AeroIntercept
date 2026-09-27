import json

import numpy as np
import pytest
import torch

from aerointercept.config import load_config
from aerointercept.end_to_end.optimization import keep_actor_batch_norm_eval
from aerointercept.end_to_end.policy import EndToEndActorCritic
from aerointercept.training import train_e2e_bc
from aerointercept.end_to_end.data import load_episode_split


def test_near_weighting_uses_measured_distance_and_preserves_visibility_padding_masks():
    batch = {"mask": torch.tensor([[1., 1., 1., 0.]]),
             "confidence": torch.tensor([[1., 1., 0., 1.]]),
             "center_distance_m": torch.tensor([[.5, 3., .4, .3]])}
    cfg = {"visible_action_only": True, "near_action_weight": 8., "near_action_radius_m": 2.}
    weights = train_e2e_bc.action_supervision_weights(batch, cfg)
    torch.testing.assert_close(weights, torch.tensor([[8., 1., 0., 0.]]))
    torch.testing.assert_close(batch["mask"], torch.tensor([[1., 1., 1., 0.]]))
    with pytest.raises(ValueError, match="measured"):
        train_e2e_bc.action_supervision_weights({"mask": batch["mask"], "confidence": batch["confidence"]}, cfg)
    with pytest.raises(ValueError, match="finite and positive"):
        train_e2e_bc.action_supervision_weights(batch, {**cfg, "near_action_weight": float('nan')})


def test_weighted_epoch_loss_is_independent_of_batch_partition():
    from aerointercept.config import DotDict

    class FixedActor(torch.nn.Module):
        def forward(self, frames, own):
            n = len(frames)
            return (torch.zeros(n, 4), torch.zeros(n, 3), torch.zeros(n),
                    torch.zeros(n), torch.ones(n, 1, 1))

    batch = {"frames": torch.zeros(1, 2, 2, 3, 8, 8, dtype=torch.uint8),
             "actions": torch.tensor([[[.4]*4, [.1]*4]]), "future_position": torch.zeros(1, 2, 3),
             "collision_risk": torch.zeros(1, 2), "confidence": torch.ones(1, 2),
             "mask": torch.ones(1, 2), "center_distance_m": torch.tensor([[.5, 3.]])}
    cfg = DotDict({**load_config().end_to_end.auxiliary, "near_action_weight": 8.})
    split = [{key: value[:, i:i+1] for key, value in batch.items()} for i in range(2)]
    combined = train_e2e_bc.run_epoch(FixedActor(), [batch], "cpu", cfg)
    separate = train_e2e_bc.run_epoch(FixedActor(), split, "cpu", cfg)
    assert combined["action"] == pytest.approx((8 * .4**2 + .1**2) / 9)
    assert combined == pytest.approx(separate)


def test_explicit_split_preserves_replay_training_assignment(tmp_path):
    replay, correction, held_out = [tmp_path / name for name in ("old.npz", "new.npz", "val.npz")]
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["old.npz", "new.npz"], "validation": ["val.npz"]}))
    assert load_episode_split([held_out, correction, replay], split) == ([replay, correction], [held_out])


@pytest.mark.parametrize("record", [
    {"train": ["old.npz"], "validation": ["old.npz"]},
    {"train": ["old.npz", "old.npz"], "validation": ["val.npz"]},
    {"train": ["old.npz"], "validation": []},
    {"train": ["old.npz"], "validation": ["../val.npz"]},
    {"train": ["old.npz"], "validation": ["val.npz", "unknown.npz"]},
    {"train": ["old.npz"], "validation": [5]},
])
def test_explicit_split_rejects_leakage_omissions_and_unknown_files(tmp_path, record):
    split = tmp_path / "split.json"
    split.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_episode_split([tmp_path / "old.npz", tmp_path / "val.npz"], split)


def test_corrective_batch_norm_keeps_statistics_but_learns_affine_parameters():
    actor = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Dropout(.1)).train()
    keep_actor_batch_norm_eval(actor)
    before = actor[0].running_mean.clone()
    variance = actor[0].running_var.clone()
    optimizer = torch.optim.SGD(actor.parameters(), lr=.01)
    initial_weight = actor[0].weight.detach().clone()
    actor(torch.ones(4, 3, 8, 8) * 7).square().mean().backward()
    optimizer.step()
    torch.testing.assert_close(actor[0].running_mean, before)
    torch.testing.assert_close(actor[0].running_var, variance)
    assert actor[0].num_batches_tracked == 0
    assert not torch.equal(actor[0].weight, initial_weight)
    assert actor[1].training


@pytest.mark.parametrize("fine_tuned_loss,expected_epoch", [(.3, 0), (.05, 1)])
@pytest.mark.parametrize("temporal", [False, True])
def test_warm_start_competes_with_fine_tuned_checkpoints(tmp_path, monkeypatch,
                                                      fine_tuned_loss, expected_epoch, temporal):
    cfg = load_config()
    cfg["end_to_end"]["render"].update(image_width=8, image_height=8)
    cfg["end_to_end"]["bc"].update(num_workers=0, backbone_freeze_epochs=0)
    episodes = tmp_path / "data" / "episodes"
    episodes.mkdir(parents=True)
    for index in range(2):
        np.savez(episodes / f"episode_{index:06d}.npz",
                 frames=np.zeros((2, 3, 8, 8), np.uint8),
                 actions=np.zeros((2, 4), np.float32),
                 future_position=np.zeros((2, 3), np.float32),
                 collision_risk=np.zeros(2, np.float32), confidence=np.ones(2, np.float32))
    (episodes.parent / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "history_frames": 2, "image_width": 8, "image_height": 8}))
    source = EndToEndActorCritic(cfg.end_to_end.model)
    initial = tmp_path / "initial.pt"
    torch.save({"model": source.state_dict(), "model_config": dict(cfg.end_to_end.model)}, initial)
    if temporal:
        cfg["end_to_end"]["model"]["temporal_memory_steps"] = 2
    output = tmp_path / "best.pt"
    trained = False

    def run_epoch(actor, loader, device, auxiliary_cfg, optimizer=None, **kwargs):
        nonlocal trained
        if optimizer is not None:
            with torch.no_grad():
                actor.action_head[-1].bias.add_(1.)
            trained = True
        value = fine_tuned_loss if trained else .1
        return {name: value for name in train_e2e_bc.Losses.__annotations__}

    monkeypatch.setattr(train_e2e_bc, "load_config", lambda _: cfg)
    monkeypatch.setattr(train_e2e_bc, "run_epoch", run_epoch)
    init_flag = "--init-temporal-checkpoint" if temporal else "--init-checkpoint"
    monkeypatch.setattr("sys.argv", ["train", "--backend", "legacy", "--device", "cpu",
                                   "--data", str(episodes.parent), init_flag, str(initial),
                                   "--out", str(output), "--epochs", "1", "--selection-metric", "action",
                                   "--sequence-length", "2"])
    train_e2e_bc.main()
    best = torch.load(output, weights_only=False)
    assert best["epoch"] == expected_epoch
    assert best["validation_loss"] == min(.1, fine_tuned_loss)
    torch.testing.assert_close(best["model"]["actor.action_head.2.bias"],
                               source.actor.action_head[-1].bias + expected_epoch)
    history = [json.loads(line) for line in output.with_suffix(".history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in history] == [0, 1]
    assert history[0]["train"] == {}
