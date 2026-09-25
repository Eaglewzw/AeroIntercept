"""Evaluation statistics that retain episode evidence and missing measurements."""

from __future__ import annotations

import math


def acceptance_result(records: list[dict], required_modes: tuple[str, ...],
                      reset_recoveries: list[dict] | None = None) -> dict:
    """Fixed engineering gate; a small pilot never counts as acceptance."""
    counts = {mode: [r for r in records if r.get("mode") == mode] for mode in required_modes}
    rates = {mode: sum(r["outcome"] == "hit" for r in rows) / len(rows) if rows else 0.
             for mode, rows in counts.items()}
    total_rate = sum(r["outcome"] == "hit" for r in records) / len(records) if records else 0.
    checks = {
        "no_reset_failures": not bool(reset_recoveries),
        "at_least_20_episodes_per_mode": all(len(rows) >= 20 for rows in counts.values()),
        "overall_success_at_least_90_percent": total_rate >= .9,
        "each_mode_success_at_least_80_percent": all(rate >= .8 for rate in rates.values()),
        "zero_contacts": bool(records) and all(r.get("contact_count") == 0 for r in records),
        "verified_body_center_reference": bool(records) and all(
            r.get("reset", {}).get("physical_state_source") == "gazebo_base_link_center_enu_to_ned_v2"
            for r in records),
        "measured_10m_resets": bool(records) and all(
            abs(r.get("reset", {}).get("target_distance_m", float("inf")) - 10.) <= .2
            for r in records),
        "measured_success_radius": all(
            r.get("rendezvous_center_distance_m") is not None
            and 0 <= r["rendezvous_center_distance_m"] <= .5
            for r in records if r["outcome"] == "hit"),
        "measured_success_speed_and_hold": all(
            r.get("rendezvous_relative_speed_mps") is not None
            and 0 <= r["rendezvous_relative_speed_mps"] <= .5
            and r.get("rendezvous_held_seconds", 0.) >= .3
            for r in records if r["outcome"] == "hit"),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "episodes_per_mode": {mode: len(rows) for mode, rows in counts.items()},
            "success_rate_per_mode": rates}


def wilson_interval(successes: int, episodes: int) -> list[float]:
    """Two-sided 95% Wilson score interval for a binomial rate."""
    if not 0 <= successes <= episodes or episodes < 1:
        raise ValueError("require 0 <= successes <= episodes and episodes > 0")
    z = 1.959963984540054
    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def mean_measured(records: list[dict], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"non-finite evaluation measurement: {key}")
    return sum(values) / len(values) if values else None


def summarize_episodes(records: list[dict]) -> dict:
    if not records:
        raise ValueError("cannot report an empty evaluation")
    outcomes = [record["outcome"] for record in records]
    successes = sum(outcome == "hit" for outcome in outcomes)
    successful_records = [record for record in records if record["outcome"] == "hit"]
    return {
        "episodes": len(records),
        "hit_rate": successes / len(records),
        "hit_rate_95ci": wilson_interval(successes, len(records)),
        "fov_lost_rate": outcomes.count("fov_lost") / len(records),
        "ground_collision_rate": outcomes.count("ground") / len(records),
        "contact_rate": outcomes.count("contact") / len(records),
        "simulator_error_rate": outcomes.count("simulator_error") / len(records),
        "mean_reward": mean_measured(records, "episode_reward"),
        "mean_minimum_distance": mean_measured(records, "minimum_distance"),
        "mean_episode_length": mean_measured(records, "episode_length"),
        "mean_episode_wall_seconds": mean_measured(records, "episode_wall_seconds"),
        "mean_episode_simulation_seconds": mean_measured(records, "episode_simulation_seconds"),
        "mean_success_simulation_seconds": mean_measured(successful_records, "episode_simulation_seconds"),
        "simulation_time_measured_episodes": sum(
            record.get("episode_simulation_seconds") is not None for record in records
        ),
        "outcomes": outcomes,
        "episode_records": records,
    }
