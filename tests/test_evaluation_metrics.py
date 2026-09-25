import pytest

from aerointercept.gazebo.environment import image_elapsed_seconds
from aerointercept.gazebo.evaluation_metrics import summarize_episodes, wilson_interval, acceptance_result


def test_acceptance_requires_complete_evidence_and_rejects_contact():
    record = {"mode": "circle", "outcome": "hit", "contact_count": 0,
              "reset": {"target_distance_m": 10., "physical_state_source": "gazebo_base_link_center_enu_to_ned_v2"},
              "rendezvous_center_distance_m": .49, "rendezvous_relative_speed_mps": .2,
              "rendezvous_held_seconds": .32}
    assert not acceptance_result([record], ("circle",))["passed"]
    records = [dict(record) for _ in range(20)]
    assert acceptance_result(records, ("circle",))["passed"]
    assert not acceptance_result(records, ("circle",), [{"error": "reset crashed"}])["passed"]
    wrong_reference = [{**r, "reset": {"target_distance_m": 10.}} for r in records]
    assert not acceptance_result(wrong_reference, ("circle",))["passed"]
    wrong_speed = [{**r, "rendezvous_relative_speed_mps": .6} for r in records]
    assert not acceptance_result(wrong_speed, ("circle",))["passed"]
    records[-1] = {**record, "outcome": "contact", "contact_count": 1}
    assert not acceptance_result(records, ("circle",))["passed"]
    assert not acceptance_result([], ("circle",))["passed"]


def test_sensor_time_does_not_assume_nominal_control_frequency():
    assert image_elapsed_seconds(1_000_000_000, 4_250_000_000) == 3.25
    assert image_elapsed_seconds(None, 100) is None
    assert image_elapsed_seconds(100, 90) is None


def test_success_time_excludes_failed_episodes_and_preserves_unknown_clocks():
    records = [
        {"outcome": "hit", "episode_simulation_seconds": 2.0},
        {"outcome": "timeout", "episode_simulation_seconds": 20.0},
        {"outcome": "hit", "episode_simulation_seconds": None},
    ]
    report = summarize_episodes(records)
    assert report["mean_success_simulation_seconds"] == 2.0
    assert report["mean_episode_simulation_seconds"] == 11.0
    assert report["simulation_time_measured_episodes"] == 2
    assert report["hit_rate"] == pytest.approx(2 / 3)
    assert report["episode_records"] == records
    assert summarize_episodes([{"outcome": "timeout"}])["mean_success_simulation_seconds"] is None


def test_small_sample_confidence_interval_is_not_perfect_certainty():
    low, high = wilson_interval(2, 2)
    assert low == pytest.approx(0.34238, abs=1e-5)
    assert high == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        summarize_episodes([])
