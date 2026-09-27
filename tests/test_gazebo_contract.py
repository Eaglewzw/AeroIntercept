"""The interception contract is enforced on configuration and on recorded weights."""

import copy
from pathlib import Path

import pytest
import yaml

from aerointercept.gazebo.checkpoint import validate_task_checkpoint
from aerointercept.gazebo.config import load_gazebo_config


TASK_CONFIGS = ("configs/gazebo_e2e.yaml", "configs/gazebo_feedback.yaml",
                "configs/gazebo_spatial.yaml")


def _config(tmp_path, **task):
    """Write one shipped configuration with a modified task block."""
    overlay = yaml.safe_load(Path(TASK_CONFIGS[0]).read_text())
    overlay["gazebo"]["task"].update(task)
    path = tmp_path/"gazebo.yaml"
    path.write_text(yaml.safe_dump(overlay))
    return load_gazebo_config(path)


@pytest.mark.parametrize("config", TASK_CONFIGS)
def test_every_shipped_configuration_declares_the_contact_intercept_task(config):
    task = load_gazebo_config(config).gazebo.task
    assert task.task_version == "contact_intercept_v1"
    assert task.require_contact_monitor is True
    assert task.center_reference == "gazebo_base_link_center_enu_to_ned_v2"


@pytest.mark.parametrize("task,match", [
    ({"task_version": "noncontact_rendezvous_v1"}, "retired"),
    ({"task_version": "contact_intercept_v2"}, "task_version must be"),
    ({"center_reference": "model_origin"}, "base_link center"),
    ({"require_contact_monitor": False}, "contact monitor"),
    ({"hit_radius": 0.}, "finite and positive"),
])
def test_configuration_without_a_complete_intercept_contract_is_rejected(tmp_path, task, match):
    """A silently weakened task would invalidate every recorded episode."""
    with pytest.raises(ValueError, match=match):
        _config(tmp_path, **task)


def test_checkpoint_from_before_the_migration_stays_loadable():
    cfg = load_gazebo_config()
    legacy = {**cfg.gazebo.task, "task_version": "noncontact_rendezvous_v1",
              "rendezvous_max_relative_speed_mps": .5, "rendezvous_hold_seconds": .3,
              "rendezvous_settle_seconds": 1.}
    validate_task_checkpoint({"task_config": legacy}, cfg)
    validate_task_checkpoint({"config": {"gazebo": {"task": dict(cfg.gazebo.task)}}}, cfg)


@pytest.mark.parametrize("task", [None, {}, {"hit_radius": .9},
                                  {"center_reference": "model_origin"}])
def test_checkpoint_with_a_different_interception_task_is_rejected(task):
    cfg = load_gazebo_config()
    checkpoint = {} if task is None else {"task_config": copy.deepcopy(task)}
    with pytest.raises(ValueError, match="train with new data"):
        validate_task_checkpoint(checkpoint, cfg)
