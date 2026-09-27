"""Contacts are charged to the vehicle that actually hit something."""

import os
import threading

import pytest

# The generated Gazebo messages in /usr/lib/python3/dist-packages predate the
# installed protobuf runtime and only load through its pure-Python parser.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
try:
    from aerointercept.gazebo.simulator_truth import SimulatorTruth
except Exception as exc:  # the Gazebo bindings need the system interpreter
    pytest.skip(f"Gazebo contact monitor bindings unavailable: {exc}",
                allow_module_level=True)


class _Body:
    def __init__(self, name):
        self.name = name


class _Contact:
    def __init__(self, first, second):
        self.collision1 = _Body(first)
        self.collision2 = _Body(second)


class _Contacts:
    def __init__(self, *pairs):
        self.contact = [_Contact(*pair) for pair in pairs]


def _monitor():
    truth = object.__new__(SimulatorTruth)
    truth.condition = threading.Condition()
    truth.contact_seen = False
    truth.contact_count = 0
    truth.last_contact = None
    truth.target_scenery_contact_count = 0
    truth.last_target_scenery_contact = None
    truth._reported_contact_pairs = set()
    return truth


def test_target_hitting_the_scenery_never_ends_the_interceptor_episode():
    truth = _monitor()
    truth.on_contacts(_Contacts(("tree_trunk::link::collision", "x500_2::base_link::base_link_collision_0")))
    assert truth.contact_count == 0
    assert truth.last_contact is None
    assert truth.target_scenery_contact_count == 1
    assert truth.last_target_scenery_contact == [
        "tree_trunk::link::collision", "x500_2::base_link::base_link_collision_0"]
    # The monitor itself is alive, which is a separate fact from the collision.
    assert truth.contact_seen


def test_interceptor_contacts_are_recorded_including_the_target_touch():
    truth = _monitor()
    truth.on_contacts(_Contacts(
        ("x500_depth_1::base_link::base_link_collision_0", "x500_2::base_link::base_link_collision_4"),
        ("ground_plane::link::collision", "x500_depth_1::rotor_2::rotor_2_collision"),
    ))
    assert truth.contact_count == 2
    assert truth.last_contact == [
        "ground_plane::link::collision", "x500_depth_1::rotor_2::rotor_2_collision"]
    assert truth.target_scenery_contact_count == 0


def test_contacts_between_other_models_are_ignored():
    truth = _monitor()
    truth.on_contacts(_Contacts(("tree_trunk::link::collision", "bench::link::collision")))
    assert (truth.contact_count, truth.target_scenery_contact_count) == (0, 0)
