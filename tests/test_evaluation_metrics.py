import pytest

from aerointercept.gazebo.environment import image_elapsed_seconds
from aerointercept.gazebo.evaluation_metrics import summarize_episodes, wilson_interval, acceptance_result


def _record(**overrides):
    return {"mode": "circle", "outcome": "hit", "contact_count": 1, "target_contact": True,
            "minimum_distance": .49,
            "reset": {"target_distance_m": 10., "physical_state_source": "gazebo_base_link_center_enu_to_ned_v2"},
            **overrides}


def test_acceptance_requires_complete_evidence_and_measured_success():
    assert not acceptance_result([_record()], ("circle",))["passed"]
    records = [_record() for _ in range(20)]
    assert acceptance_result(records, ("circle",))["passed"]
    # A pilot never counts, and every mode must stand on its own.
    assert not acceptance_result(records, ("circle", "figure_eight"))["passed"]
    assert not acceptance_result(records, ("circle",), [{"error": "reset crashed"}])["passed"]
    wrong_reference = [{**r, "reset": {"target_distance_m": 10.}} for r in records]
    assert not acceptance_result(wrong_reference, ("circle",))["passed"]
    wrong_distance = [{**r, "reset": {"target_distance_m": 8., "physical_state_source":
                                      "gazebo_base_link_center_enu_to_ned_v2"}} for r in records]
    assert not acceptance_result(wrong_distance, ("circle",))["passed"]
    assert not acceptance_result([], ("circle",))["passed"]


def test_acceptance_rejects_a_hit_without_measured_success_evidence():
    """A recorded hit must carry its own evidence: contact or measured radius."""
    records = [_record() for _ in range(20)]
    # Claimed by outcome alone: no contact and the measured pass stayed outside.
    unmeasured = [{**r, "target_contact": False, "minimum_distance": .6} for r in records]
    assert not acceptance_result(unmeasured, ("circle",))["passed"]
    # A target contact is accepted on its own, even for a missed pass.
    contacted = [{**r, "target_contact": True, "minimum_distance": .6} for r in records]
    assert acceptance_result(contacted, ("circle",))["passed"]
    # Missing measurement is not evidence either.
    assert not acceptance_result([{**r, "minimum_distance": None} for r in unmeasured],
                                 ("circle",))["passed"]
    # The configured radius, not a hardcoded one, decides the boundary.
    borderline = [{**r, "target_contact": False, "minimum_distance": .6} for r in records]
    assert acceptance_result(borderline, ("circle",), hit_radius=.7)["passed"]


def test_contact_with_something_other_than_the_target_is_a_failed_episode():
    records = [_record() for _ in range(20)]
    records[-1] = _record(outcome="contact", target_contact=False, contact_count=1)
    result = acceptance_result(records, ("circle",))
    assert result["success_rate_per_mode"]["circle"] == pytest.approx(19/20)
    assert result["passed"]
    # Enough of them and the measured success rate falls through the gate.
    records[17:19] = [_record(outcome="contact", target_contact=False, contact_count=1)]*2
    assert not acceptance_result(records, ("circle",))["passed"]


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
