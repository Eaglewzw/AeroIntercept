"""Explicit ENU/FLU (Gazebo) and NED/FRD (PX4) coordinate transforms."""

import math
import numpy as np


ENU_TO_NED = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
FLU_TO_FRD = np.diag([1., -1., -1.])
# Resolved x500 SDF: base_link (the main body's inertial origin) is
# 0.24 m above the top-level model origin. Camera is at z=0.26078
# relative to that model, hence only 0.02078 m above the body center.
BODY_CENTER_OFFSET_FRD = np.array([0., 0., -.24])
CAMERA_OFFSET_FRD = np.array([0.13233, 0.0, -0.02078])
PHYSICAL_STATE_SOURCE = "gazebo_base_link_center_enu_to_ned_v2"


def enu_to_ned(position):
    return ENU_TO_NED @ np.asarray(position, dtype=np.float64)


def quaternion_matrix(wxyz):
    q = np.asarray(wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("invalid pose quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def body_frd_to_ned(gazebo_quaternion_wxyz):
    return ENU_TO_NED @ quaternion_matrix(gazebo_quaternion_wxyz) @ FLU_TO_FRD


def model_origin_to_body_center(position_ned, gazebo_quaternion_wxyz):
    return np.asarray(position_ned)+body_frd_to_ned(gazebo_quaternion_wxyz)@BODY_CENTER_OFFSET_FRD


def camera_relative_frd(interceptor_ned, target_ned, gazebo_quaternion_wxyz):
    rotation = body_frd_to_ned(gazebo_quaternion_wxyz)
    return rotation.T @ (np.asarray(target_ned) - interceptor_ned) - CAMERA_OFFSET_FRD


def interpolate_pose(a, b, timestamp_ns):
    if not a["timestamp_ns"] <= timestamp_ns <= b["timestamp_ns"] or b["timestamp_ns"] <= a["timestamp_ns"]:
        raise ValueError("interpolation requires an ordered bracket around the timestamp")
    fraction = (timestamp_ns-a["timestamp_ns"])/(b["timestamp_ns"]-a["timestamp_ns"])
    qa, qb = np.asarray(a["quaternion_enu_wxyz"]), np.asarray(b["quaternion_enu_wxyz"])
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = qa*(1-fraction)+qb*fraction
    q /= np.linalg.norm(q)
    rotation = body_frd_to_ned(q)
    return {
        "position": np.asarray(a["position"])*(1-fraction)+np.asarray(b["position"])*fraction,
        "velocity": np.asarray(a["velocity"])*(1-fraction)+np.asarray(b["velocity"])*fraction,
        "quaternion_enu_wxyz": q.tolist(), "timestamp_ns": timestamp_ns,
        "yaw": math.atan2(rotation[1, 0], rotation[0, 0]),
    }
