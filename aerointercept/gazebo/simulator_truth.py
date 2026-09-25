"""System-Python Gazebo pose and contact monitor; never an Actor input."""

import math
import json
from collections import defaultdict, deque

import numpy as np
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.contacts_pb2 import Contacts
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean

from aerointercept.gazebo.frames import body_frd_to_ned, enu_to_ned, interpolate_pose, model_origin_to_body_center
from aerointercept.gazebo.world_control import request_pause_state


class SimulatorTruth:
    def __init__(self, condition, world="aerointercept_park"):
        self.condition = condition
        self.node = Node()
        self.poses = {}
        self.histories = defaultdict(lambda: deque(maxlen=64))
        self.contact_seen = False
        self.contact_count = 0
        self.last_contact = None
        self._reported_contact_pairs = set()
        self.world = world
        self.node.subscribe(Pose_V, "/aerointercept/model_poses", self.on_poses)
        self.node.subscribe(Contacts, "/aerointercept/contacts", self.on_contacts)

    def on_poses(self, message):
        with self.condition:
            for pose in message.pose:
                if pose.name not in ("x500_depth_1", "x500_2"):
                    continue
                stamp_value = pose.header.stamp if pose.HasField("header") else message.header.stamp
                stamp = stamp_value.sec*1_000_000_000 + stamp_value.nsec
                position = enu_to_ned([pose.position.x, pose.position.y, pose.position.z])
                q = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
                position = model_origin_to_body_center(position, q)
                rotation = body_frd_to_ned(q)
                previous = self.poses.get(pose.name)
                if previous and stamp <= previous["timestamp_ns"]:
                    continue
                velocity = np.zeros(3)
                if previous and stamp > previous["timestamp_ns"]:
                    velocity = (position-previous["position"]) / ((stamp-previous["timestamp_ns"])*1e-9)
                self.poses[pose.name] = {
                    "position": position, "velocity": velocity,
                    "quaternion_enu_wxyz": q, "timestamp_ns": stamp,
                    "yaw": math.atan2(rotation[1, 0], rotation[0, 0]),
                }
                self.histories[pose.name].append(self.poses[pose.name])
            self.condition.notify_all()

    def on_contacts(self, message):
        with self.condition:
            self.contact_seen = True
            for contact in message.contact:
                a, b = contact.collision1.name, contact.collision2.name
                names = ("x500_depth_1", "x500_2")
                owners_a = {name for name in names if name in a}
                owners_b = {name for name in names if name in b}
                if not (owners_a or owners_b) or owners_a & owners_b:
                    continue
                self.contact_count += 1
                self.last_contact = [a, b]
                pair = tuple(sorted((a, b)))
                if pair not in self._reported_contact_pairs:
                    self._reported_contact_pairs.add(pair)
                    print("GAZEBO_CONTACT="+json.dumps({"pair": pair}), flush=True)
            self.condition.notify_all()

    @property
    def ready(self):
        return len(self.poses) == 2

    def ready_at(self, timestamp_ns):
        return self.ready and all(history[0]["timestamp_ns"] <= timestamp_ns <= history[-1]["timestamp_ns"]
                                  for history in self.histories.values())

    def at_timestamp(self, timestamp_ns):
        """Align both model centers and camera exposure to one simulation time."""
        if not self.ready_at(timestamp_ns):
            raise ValueError("pose history does not bracket the camera timestamp")
        result = {}
        for name, history in self.histories.items():
            for a, b in zip(history, list(history)[1:]):
                if a["timestamp_ns"] <= timestamp_ns <= b["timestamp_ns"]:
                    result[name] = interpolate_pose(a, b, timestamp_ns)
                    break
            else:
                result[name] = dict(history[0])
        return result

    def set_paused(self, paused):
        def request(value):
            response, reply = self.node.request(f"/world/{self.world}/control",
                WorldControl(pause=value), WorldControl, Boolean, 2000)
            return response, bool(reply.data) if response else False
        request_pause_state(request, paused)
