import numpy as np
import pytest

from aerointercept.gazebo.config import load_gazebo_config
from aerointercept.gazebo.frames import enu_to_ned, body_frd_to_ned, camera_relative_frd, interpolate_pose
from aerointercept.gazebo.frames import model_origin_to_body_center
from aerointercept.gazebo.scenarios import Scenario, MODES
from aerointercept.gazebo.task_logic import termination_flags
from aerointercept.gazebo.task_logic import target_visibility
from aerointercept.gazebo.task_logic import task_contract
from aerointercept.gazebo.expert import GazeboExpertController
from aerointercept.end_to_end.actions import decode_action


def test_enu_origin_and_camera_forward_are_not_conflated():
    assert np.array_equal(enu_to_ned([10., 0., 2.]), [0., 10., -2.])
    assert np.allclose(body_frd_to_ned([1, 0, 0, 0])[:, 0], [0, 1, 0])
    relative = camera_relative_frd(np.zeros(3), np.array([0., 10., 0.]), [1, 0, 0, 0])
    assert relative[0] > 9 and relative[1] == 0
    assert relative[2] == pytest.approx(.02078)


def test_body_center_rotates_with_model_instead_of_using_world_height_offset():
    np.testing.assert_allclose(model_origin_to_body_center([1, 2, -6], [1, 0, 0, 0]), [1, 2, -6.24])
    q = [np.cos(np.pi/4), np.sin(np.pi/4), 0, 0]
    np.testing.assert_allclose(model_origin_to_body_center([1, 2, -6], q), [.76, 2, -6], atol=1e-8)


def test_partial_vehicle_visibility_is_distinct_from_center_visibility():
    assert not target_visibility([.4, 0, -.2], 1.2, .74)[0]
    assert target_visibility([.4, 0, -.2], 1.2, .74, .4)[0]
    assert not target_visibility([2, 5, 0], 1.2, .74, .4)[0]
    assert not target_visibility([-1, 0, 0], 1.2, .74, .4)[0]


def test_nominal_formation_meets_radius_with_center_inside_camera():
    cfg = load_gazebo_config()
    relative = -np.asarray(cfg.gazebo.expert.offset_ned)
    assert np.linalg.norm(relative) < cfg.gazebo.task.hit_radius
    yaw_north = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]
    camera = camera_relative_frd(np.zeros(3), relative, yaw_north)
    assert target_visibility(camera, cfg.gazebo.camera.horizontal_fov, cfg.gazebo.camera.vertical_fov)[0]


def test_pose_interpolation_aligns_exposure_and_handles_quaternion_sign():
    a = {"position": [0., 0., 0.], "velocity": [1., 0., 0.],
         "quaternion_enu_wxyz": [1., 0., 0., 0.], "timestamp_ns": 0}
    b = {**a, "position": [1., 0., 0.], "quaternion_enu_wxyz": [-1., 0., 0., 0.],
         "timestamp_ns": 1_000_000_000}
    state = interpolate_pose(a, b, 250_000_000)
    assert np.allclose(state["position"], [.25, 0., 0.])
    assert np.allclose(state["quaternion_enu_wxyz"], [1., 0., 0., 0.])
    with pytest.raises(ValueError):
        interpolate_pose(a, b, 2_000_000_000)


@pytest.mark.parametrize("mode", MODES)
def test_scenarios_are_repeatable_bounded_and_start_at_ten_metres(mode):
    a = Scenario.sample(mode, 31)
    b = Scenario.sample(mode, 31)
    assert np.linalg.norm(a.position(0)-[0, 0, -6]) == pytest.approx(10.)
    points = np.stack([a.position(t) for t in np.linspace(0, 300, 1000)])
    assert np.isfinite(points).all()
    assert np.max(np.linalg.norm(points[:, :2], axis=1)) < 25
    assert np.array_equal(a.position(17.5), b.position(17.5))
    assert a.metadata() != Scenario.sample(mode, 32).metadata()


def test_cooperative_expert_matches_velocity_at_formation_offset():
    cfg = load_gazebo_config()
    expert = GazeboExpertController(cfg.gazebo.expert, cfg.gazebo.action, cfg.gazebo.camera)
    target = np.array([10., 0., -6.])
    state = {"target_position": target,
             "interceptor_position": target+cfg.gazebo.expert.offset_ned,
             "target_velocity": [.2, .1, 0.], "interceptor_velocity": [.2, .1, 0.],
             "interceptor_yaw": 0.}
    # Let the existing command filter settle.
    for _ in range(10):
        action = expert.action(state)
    decoded = decode_action(action, 0., velocity_max=cfg.gazebo.action.velocity_max,
                            yaw_rate_max=cfg.gazebo.action.yaw_rate_max)
    assert np.allclose(decoded.ned_velocity, state["target_velocity"], atol=1e-5)


def test_cooperative_expert_keeps_position_correction_at_close_range():
    cfg = load_gazebo_config()
    expert = GazeboExpertController(cfg.gazebo.expert, cfg.gazebo.action, cfg.gazebo.camera)
    state = {"target_position": [.6, 0., -6.],
             "interceptor_position": [0., 0., -6.],
             "target_velocity": [0., 0., 0.],
             "interceptor_velocity": [0., 0., 0.],
             "interceptor_yaw": 0.}
    action = expert.action(state)
    decoded = decode_action(action, 0., velocity_max=cfg.gazebo.action.velocity_max,
                            yaw_rate_max=cfg.gazebo.action.yaw_rate_max)
    expected = (1.-cfg.gazebo.expert.action_smoothing)*cfg.gazebo.expert.position_gain*.14
    assert decoded.north == pytest.approx(expected)


def _flags(cfg, *, distance, contact=False, target_contact=False, **overrides):
    """Judge one step of the current task; the retired keys take no part in it."""
    arguments = dict(
        step_minimum_distance=distance, lost_count=0,
        interceptor_position=[0, 0, -6], invalid=False, episode_step=1,
        cfg=cfg.gazebo.task, contact=contact, target_contact=target_contact,
    )
    arguments.update(overrides)
    return termination_flags(**arguments)


@pytest.mark.parametrize("distance,contact,target_contact,success,physical_contact", [
    (.49, False, False, True, False),
    (.5, False, False, True, False),
    (.51, False, False, False, False),
    (.7, True, True, True, False),
    (.49, True, False, False, True),
])
def test_contact_intercept_success_is_radius_or_target_contact(
        distance, contact, target_contact, success, physical_contact):
    """Relative speed and hold time are not conditions of this task."""
    cfg = load_gazebo_config()
    flags = _flags(cfg, distance=distance, contact=contact, target_contact=target_contact)
    assert flags["hit"] is success
    # Contact with the target is the success event, not a collision penalty.
    assert flags["contact"] is physical_contact
    assert flags["terminated"] is (success or physical_contact)


def test_hit_radius_boundary_is_inclusive():
    cfg = load_gazebo_config()
    radius = float(cfg.gazebo.task.hit_radius)
    assert _flags(cfg, distance=radius)["hit"]
    assert not _flags(cfg, distance=radius + 1e-3)["hit"]


def test_target_contact_succeeds_even_when_the_pass_missed_the_radius():
    cfg = load_gazebo_config()
    flags = _flags(cfg, distance=3., contact=True, target_contact=True)
    assert flags["hit"] and not flags["contact"] and flags["terminated"]


@pytest.mark.parametrize("failure,overrides", [
    ("invalid", {"invalid": True}),
    ("ground", {"interceptor_position": [0, 0, 0]}),
    ("out_of_bounds", {"interceptor_position": [0, 60, -6]}),
])
def test_distance_never_overrides_a_physical_failure(failure, overrides):
    cfg = load_gazebo_config()
    flags = _flags(cfg, distance=.01, target_contact=True, **overrides)
    assert flags[failure] and not flags["hit"] and flags["terminated"]


def test_losing_the_target_ends_the_episode_without_a_hit():
    cfg = load_gazebo_config()
    flags = _flags(cfg, distance=9., lost_count=int(cfg.gazebo.task.lost_steps))
    assert flags["fov_lost"] and not flags["hit"] and flags["terminated"]


def test_terminal_frames_of_a_hit_are_not_a_camera_failure():
    """The last frames of an intercept fill or leave the image."""
    cfg = load_gazebo_config()
    assert not _flags(cfg, distance=9., lost_count=10 ** 6)["hit"]
    assert _flags(cfg, distance=.01, lost_count=10 ** 6)["hit"]


def test_timed_out_is_a_truncation_not_a_termination():
    cfg = load_gazebo_config()
    flags = _flags(cfg, distance=9., episode_step=cfg.gazebo.task.episode_max_steps)
    assert flags["timed_out"] and not flags["terminated"]


def test_retired_rendezvous_keys_and_version_label_do_not_change_the_contract():
    """Recordings made before the migration stay comparable to this task."""
    cfg = load_gazebo_config()
    legacy = {
        **cfg.gazebo.task, "task_version": "noncontact_rendezvous_v1",
        "rendezvous_max_relative_speed_mps": .5, "rendezvous_hold_seconds": .3,
        "rendezvous_settle_seconds": 1.,
    }
    current = task_contract(cfg.gazebo.task)
    assert current == task_contract(legacy)
    # Only the version label drops out of the current task; every other field
    # still separates two genuinely different tasks.
    assert set(current) == set(cfg.gazebo.task) - {"task_version"}
    for key, value in (("hit_radius", .9), ("center_reference", "model_origin"),
                       ("reset_target_distance_m", 5.)):
        assert task_contract({**legacy, key: value}) != current
