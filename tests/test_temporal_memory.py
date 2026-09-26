import torch

from aerointercept.config import DotDict, load_config
from aerointercept.end_to_end.policy import EndToEndActorCritic
from aerointercept.gazebo.checkpoint import load_temporal_initialization
from aerointercept.training.train_e2e_bc import compute_losses


def models():
    config = dict(load_config().end_to_end.model)
    config.update(self_state_dim=6, dropout=0.)
    source = EndToEndActorCritic(DotDict(config)).eval()
    temporal_config = {**config, "temporal_memory_steps": 4}
    temporal = EndToEndActorCritic(DotDict(temporal_config)).eval()
    load_temporal_initialization(temporal, {"model_config": config, "model": source.state_dict()}, temporal_config)
    return source, temporal


def test_temporal_migration_preserves_initial_action_and_trains_recurrence():
    torch.manual_seed(9)
    source, temporal = models()
    frames = torch.randint(0, 256, (1, 4, 2, 3, 32, 48), dtype=torch.uint8)
    own = torch.randn(1, 4, 6)
    with torch.no_grad():
        expected = source.actor(frames[:, 0], own[:, 0])[0]
        actual = temporal.actor(frames[:, 0], own[:, 0])[0]
    assert torch.allclose(actual, expected, atol=1e-6)
    with torch.no_grad():
        temporal.actor.recurrent_projection.weight.copy_(torch.eye(128))
    batch = {"frames": frames, "self_state": own, "actions": torch.ones(1, 4, 4)*.2,
             "future_position": torch.zeros(1, 4, 3), "collision_risk": torch.zeros(1, 4),
             "confidence": torch.ones(1, 4), "mask": torch.ones(1, 4)}
    loss = compute_losses(temporal.actor, batch, "cpu", load_config().end_to_end.auxiliary)
    loss.total.backward()
    assert temporal.actor.recurrent.weight_ih_l0.grad.abs().sum() > 0


def test_rolling_inference_matches_causal_windows_and_resets_after_export(tmp_path):
    torch.manual_seed(7)
    _, model = models()
    actor = model.actor
    with torch.no_grad():
        actor.recurrent_projection.weight.copy_(torch.eye(128))
    frames = torch.randint(0, 256, (1, 7, 2, 3, 32, 48), dtype=torch.uint8)
    own = torch.randn(1, 7, 6)
    scripted = torch.jit.script(actor)
    with torch.no_grad():
        for index in range(7):
            begin = max(0, index-3)
            expected = actor.forward_sequence(frames[:, begin:index+1], own[:, begin:index+1])[0][-1:]
            assert torch.allclose(scripted(frames[:, index], own[:, index])[0], expected, atol=2e-6)
        original = actor.forward_sequence(frames[:, :4], own[:, :4])[0]
        changed = frames[:, :4].clone()
        changed[:, 3] = 255-changed[:, 3]
        assert torch.allclose(actor.forward_sequence(changed, own[:, :4])[0][:3], original[:3], atol=2e-6)
        scripted.reset_memory()
        path = tmp_path/"temporal.pt"
        scripted.save(str(path))
        restored = torch.jit.load(str(path)).eval()
        expected_first = actor.forward_sequence(frames[:, :1], own[:, :1])[0]
        assert torch.allclose(restored(frames[:, 0], own[:, 0])[0], expected_first, atol=2e-6)
