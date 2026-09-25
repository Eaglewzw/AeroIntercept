import numpy as np
import pytest
import torch
import json

from aerointercept.config import DotDict, load_config
from aerointercept.end_to_end.policy import EndToEndActorCritic
from aerointercept.gazebo.checkpoint import load_visual_initialization
from aerointercept.gazebo.self_state import px4_self_state, past_self_state
from aerointercept.end_to_end.data import EpisodeSequenceDataset
from aerointercept.training.train_e2e_ppo import ImageRolloutBuffer, ppo_update


def test_px4_body_rotation_and_angular_units():
    q = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]
    result = px4_self_state(q, [0, 2, -1], [.1, -.2, .3], 1, 1)
    np.testing.assert_allclose(result, [2, 0, -1, .1, -.2, .3], atol=1e-6)
    np.testing.assert_allclose(px4_self_state(q, [1, 2, 3], [0, 0, 1], 3, 1), [1, 2, 3, 0, 0, 1])
    with pytest.raises(ValueError):
        px4_self_state(q, [1, 2, 3], [0, 0, 1], 2, 1)
    with pytest.raises(ValueError):
        px4_self_state(q, [1, np.nan, 3], [0, 0, 1], 3, 1)


def test_own_sample_is_causal_and_fresh():
    history = [(1_000_000_000, np.zeros(6)), (1_100_000_000, np.ones(6))]
    assert past_self_state(history, 1_050_000_000)[0] == 1_000_000_000
    assert past_self_state(history, 999_000_000) is None
    assert past_self_state(history, 1_400_000_000) is None


def test_dataset_requires_real_synchronized_self_state(tmp_path):
    arrays = {"frames": np.zeros((2, 3, 8, 8), np.uint8), "actions": np.zeros((2, 4), np.float32),
              "future_position": np.zeros((2, 3), np.float32), "collision_risk": np.zeros(2),
              "confidence": np.ones(2)}
    path = tmp_path / "episode.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="self_state"):
        EpisodeSequenceDataset([path], 2, self_state_dim=6)
    arrays.update(self_state=np.ones((2, 6), np.float32),
                  image_timestamp_ns=np.array([1000000000, 1100000000]),
                  self_state_timestamp_ns=np.array([990000000, 1090000000]))
    np.savez(path, **arrays)
    item = EpisodeSequenceDataset([path], 2, self_state_dim=6)[0]
    np.testing.assert_array_equal(item["self_state"], arrays["self_state"])
    arrays["self_state_timestamp_ns"][0] = 1200000000
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="future or stale"):
        EpisodeSequenceDataset([path], 2, self_state_dim=6)


def test_visual_migration_preserves_initial_policy_and_trains_sensor_fusion():
    old_cfg = load_config().end_to_end.model
    old_model = EndToEndActorCritic(old_cfg).eval()
    new_cfg = DotDict({**old_cfg, "self_state_dim": 6})
    new_model = EndToEndActorCritic(new_cfg).eval()
    load_visual_initialization(new_model, {"model_config": dict(old_cfg), "model": old_model.state_dict()}, new_cfg)
    frames = torch.randint(0, 256, (2, 2, 3, 32, 32), dtype=torch.uint8)
    own = torch.randn(2, 6)
    torch.testing.assert_close(old_model.actor(frames)[0], new_model.actor(frames, own)[0])
    with pytest.raises(ValueError):
        new_model.actor(frames)
    with pytest.raises(ValueError):
        new_model.actor(frames, torch.zeros(2, 15))
    new_model.actor(frames, own)[0].square().sum().backward()
    assert new_model.actor.sensor_fusion.weight.grad[:, -64:].abs().sum() > 0
    scripted = torch.jit.script(new_model.actor)
    torch.testing.assert_close(scripted(frames, own)[0], new_model.actor(frames, own)[0])


def test_sensor_ppo_buffer_and_update_use_measured_inputs():
    cfg = load_config()
    model_cfg = DotDict({**cfg.end_to_end.model, "self_state_dim": 6})
    model = EndToEndActorCritic(model_cfg).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    buffer = ImageRolloutBuffer(2, 1, (2, 3, 32, 32), 15, 4, self_state_dim=6)
    info = {"critic_obs": np.zeros((1, 15), np.float32), "future_position": np.zeros((1, 3), np.float32),
            "collision_risk": np.zeros(1, np.float32), "confidence": np.ones(1, np.float32)}
    own = np.array([[1., .2, 0., 0., .1, .3]], np.float32)
    for _ in range(2):
        frames = np.zeros((1, 2, 3, 32, 32), np.uint8)
        action, logp, value = model.act(torch.from_numpy(frames), torch.zeros(1, 15),
                                        self_state=torch.from_numpy(own))[:3]
        buffer.add(frames, info, action, logp, np.ones(1, np.float32), np.zeros(1, np.float32),
                   value, self_state=own)
    np.testing.assert_array_equal(buffer.self_state[0].numpy(), own)
    buffer.compute_gae(torch.zeros(1), .99, .95)
    ppo_cfg = DotDict({**cfg.end_to_end.ppo, "epochs": 1, "num_minibatches": 1})
    metrics = ppo_update(model, optimizer, buffer, ppo_cfg, cfg.end_to_end.auxiliary, torch.device("cpu"))
    assert all(np.isfinite(value) for value in metrics.values())
    assert model.actor.sensor_fusion.weight.grad is not None


def test_export_runtime_requires_sensor_sample_and_freshness(tmp_path, monkeypatch):
    from aerointercept.export_e2e import main as export_main
    from aerointercept.end_to_end.runtime import EndToEndRuntime
    cfg = load_config()
    model_cfg = DotDict({**cfg.end_to_end.model, "self_state_dim": 6})
    model = EndToEndActorCritic(model_cfg).eval()
    checkpoint = tmp_path/"source.pt"
    output = tmp_path/"policy.pt"
    torch.save({"phase": 3, "model": model.state_dict(), "model_config": dict(model_cfg)}, checkpoint)
    monkeypatch.setattr("sys.argv", ["export", "--ckpt", str(checkpoint), "--out", str(output)])
    export_main()
    metadata = json.loads((tmp_path/"policy_meta.json").read_text())
    assert metadata["policy_version"] == "rgb_px4_self_state_v1"
    assert not metadata["contains_critic"]
    runtime = EndToEndRuntime(output)
    frame = np.zeros((108, 192, 3), np.uint8)
    with pytest.raises(ValueError, match="self_state"):
        runtime.step(frame, 0.)
    with pytest.raises(ValueError, match="fresh"):
        runtime.step(frame, 0., self_state=np.zeros(6), self_state_age_seconds=.3)
    result = runtime.step(frame, 0., self_state=np.zeros(6), self_state_age_seconds=.01)
    assert result.action.shape == (4,)
