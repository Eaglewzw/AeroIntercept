import pytest
import torch
import numpy as np

from aerointercept.config import DotDict, load_config
from aerointercept.end_to_end.policy import EndToEndActorCritic
from aerointercept.gazebo.checkpoint import load_spatial_initialization
from aerointercept.training.train_e2e_bc import spatial_attention_loss
from aerointercept.training.train_e2e_bc import compute_losses
from aerointercept.gazebo.visual_supervision import camera_target_xy
from aerointercept.training.train_e2e_ppo import ImageRolloutBuffer, ppo_update


def test_projection_keeps_letterbox_pixel_aspect_and_axis_signs():
    np.testing.assert_allclose(camera_target_xy([5, 0, 0], np.pi/2), [0, 0])
    np.testing.assert_allclose(camera_target_xy([5, 5, -2.5], np.pi/2), [1, -.5])
    with pytest.raises(ValueError):
        camera_target_xy([5, np.nan, 0], 1.2)


def test_spatial_migration_preserves_outputs_and_export():
    cfg = DotDict({**load_config().end_to_end.model, "self_state_dim": 6})
    old = EndToEndActorCritic(cfg).eval()
    new_cfg = DotDict({**cfg, "spatial_coordinates": True})
    new = EndToEndActorCritic(new_cfg).eval()
    load_spatial_initialization(new, {"model_config": dict(cfg), "model": old.state_dict()}, new_cfg)
    frames = torch.randint(0, 256, (2, 2, 3, 32, 32), dtype=torch.uint8)
    own = torch.randn(2, 6)
    torch.testing.assert_close(old.actor(frames, own)[0], new.actor(frames, own)[0])
    new.actor(frames, own)[0].sum().backward()
    assert new.actor.spatial_projection.weight.grad.abs().sum() > 0
    exported = torch.jit.script(new.actor)
    torch.testing.assert_close(exported(frames, own)[0], new.actor(frames, own)[0])
    with pytest.raises(ValueError):
        load_spatial_initialization(new, {"model_config": dict(new_cfg), "model": new.state_dict()}, new_cfg)


def test_supervision_moves_attention_toward_label_and_ignores_invisible():
    logits = torch.zeros(2, 20, 20, requires_grad=True)
    xy = torch.tensor([[.65, -.25], [-.65, .25]])
    attention = logits.flatten(1).softmax(-1).view_as(logits)
    loss = spatial_attention_loss(attention, xy, torch.tensor([1., 0.]))
    loss.backward()
    assert logits.grad[0, 7, 16] < 0  # gradient descent raises the target cell
    assert logits.grad[0, 7, 3] > 0
    assert logits.grad[1].abs().sum() == 0
    invisible = spatial_attention_loss(attention.detach(), xy, torch.zeros(2))
    assert invisible == 0


def test_rgb_yaw_feedback_is_camera_calibrated_and_exportable():
    cfg = DotDict({**load_config().end_to_end.model, "self_state_dim": 6,
                   "spatial_coordinates": True})
    old = EndToEndActorCritic(cfg).eval()
    new = EndToEndActorCritic(DotDict({**cfg, "visual_yaw_gain": 2., "camera_horizontal_fov": 1.2})).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    frames = torch.randint(0, 256, (2, 2, 3, 32, 32), dtype=torch.uint8)
    own = torch.zeros(2, 6)
    raw, corrected = old.actor(frames, own), new.actor(frames, own)
    attention = raw[-1]
    x = (torch.arange(attention.shape[-1])+.5)*2/attention.shape[-1]-1
    bearing = torch.atan((attention*x[None, None, :]).sum((-2,-1))*np.tan(.6))
    torch.testing.assert_close(torch.atanh(corrected[0][:,3])-torch.atanh(raw[0][:,3]), 2*bearing,
                               atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(raw[0][:,:3], corrected[0][:,:3])
    exported = torch.jit.script(new.actor)
    torch.testing.assert_close(exported(frames, own)[0], corrected[0])


def test_bc_axis_weight_controls_vertical_error_penalty():
    class FixedActor(torch.nn.Module):
        def forward(self, frames, own):
            return (torch.tensor([[0., 0., .1, 0.]]), torch.zeros(1, 3),
                    torch.zeros(1), torch.zeros(1), torch.ones(1, 1, 1))
    batch = {"frames": torch.zeros(1, 1, 2, 3, 8, 8, dtype=torch.uint8),
             "actions": torch.zeros(1, 1, 4), "future_position": torch.zeros(1, 1, 3),
             "collision_risk": torch.zeros(1, 1), "confidence": torch.ones(1, 1),
             "mask": torch.ones(1, 1)}
    base = load_config().end_to_end.auxiliary
    weighted = DotDict({**base, "action_axis_weights": [1, 1, 16, 4]})
    before = compute_losses(FixedActor(), batch, "cpu", base).action
    after = compute_losses(FixedActor(), batch, "cpu", weighted).action
    torch.testing.assert_close(after, before*16)
    with pytest.raises(ValueError):
        compute_losses(FixedActor(), batch, "cpu", DotDict({**base, "action_axis_weights": [1, 1, -1, 1]}))


def test_ppo_retains_training_only_camera_supervision():
    cfg = load_config()
    model = EndToEndActorCritic(DotDict({**cfg.end_to_end.model, "self_state_dim": 6,
        "spatial_coordinates": True, "visual_yaw_gain": 2.})).eval()
    buffer = ImageRolloutBuffer(2, 1, (2, 3, 32, 32), 15, 4, self_state_dim=6, camera_supervision=True)
    info = {"critic_obs": np.zeros((1, 15), np.float32), "future_position": np.zeros((1, 3), np.float32),
            "collision_risk": np.zeros(1, np.float32), "confidence": np.ones(1, np.float32),
            "camera_target_xy": np.array([[.4, -.2]], np.float32), "camera_target_valid": np.ones(1, np.float32)}
    own = np.zeros((1, 6), np.float32)
    for _ in range(2):
        frames = np.zeros((1, 2, 3, 32, 32), np.uint8)
        action, logp, value = model.act(torch.from_numpy(frames), torch.zeros(1, 15),
                                      self_state=torch.from_numpy(own))[:3]
        buffer.add(frames, info, action, logp, np.ones(1, np.float32), np.zeros(1, np.float32), value, self_state=own)
    np.testing.assert_array_equal(buffer.camera_target_xy[0].numpy(), info["camera_target_xy"])
    buffer.compute_gae(torch.zeros(1), .99, .95)
    metrics = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-5), buffer,
                        DotDict({**cfg.end_to_end.ppo, "epochs": 1, "num_minibatches": 1}),
                        DotDict({**cfg.end_to_end.auxiliary, "spatial_coef": .05}), torch.device("cpu"))
    assert np.isfinite(metrics["spatial_loss"]) and metrics["spatial_loss"] > 0
    assert model.actor.encoder.attention.weight.grad.abs().sum() > 0
