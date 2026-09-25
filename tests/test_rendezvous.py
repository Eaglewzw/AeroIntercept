import numpy as np
import pytest

from aerointercept.gazebo.config import load_gazebo_config
from aerointercept.gazebo.frames import enu_to_ned, body_frd_to_ned, camera_relative_frd, interpolate_pose
from aerointercept.gazebo.frames import model_origin_to_body_center
from aerointercept.gazebo.scenarios import Scenario, MODES
from aerointercept.gazebo.task_logic import termination_flags
from aerointercept.gazebo.task_logic import target_visibility
from aerointercept.gazebo.task_logic import vertical_clearance_penalty
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


def test_clearance_reward_prefers_safe_approach_without_changing_success():
    cfg = {"vertical_clearance_weight": 1., "vertical_clearance_m": .14,
           "vertical_clearance_activation_m": 2.}
    assert vertical_clearance_penalty([.46, 0, -.14], cfg) == 0
    assert vertical_clearance_penalty([3, 0, 0], cfg) == 0
    assert vertical_clearance_penalty([.46, 0, .1], cfg) < vertical_clearance_penalty([.46, 0, 0], cfg) < 0
    assert vertical_clearance_penalty([.46, 0, 0], {}) == 0


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


@pytest.mark.parametrize("distance,speed,hold,contact,monitor,success", [
    (.5, .5, .3, False, True, True),
    (.5001, .1, .3, False, True, False),
    (.49, .5001, .3, False, True, False),
    (.49, .1, .299, False, True, False),
    (.49, .1, .3, True, True, False),
    (.49, .1, .3, False, False, False),
])
def test_noncontact_success_requires_all_conditions(distance, speed, hold, contact, monitor, success):
    cfg = load_gazebo_config()
    flags = termination_flags(step_minimum_distance=.01, lost_count=0,
        interceptor_position=[0, 0, -6], invalid=False, episode_step=1,
        cfg=cfg.gazebo.task, current_distance=distance, relative_speed=speed,
        held_seconds=hold, contact=contact, contact_monitor_ready=monitor)
    assert flags["hit"] is success
    if contact:
        assert flags["contact"] and flags["terminated"]


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
