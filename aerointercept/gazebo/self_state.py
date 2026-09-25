"""Deployable own-vehicle state; never derived from target or Gazebo poses."""
import numpy as np
from .frames import quaternion_matrix

SELF_STATE_SOURCE = "px4_vehicle_odometry_body_frd_v1"
SELF_STATE_COMPONENTS = ("vx", "vy", "vz", "roll_rate", "pitch_rate", "yaw_rate")


def px4_self_state(quaternion, velocity, angular_velocity, velocity_frame, pose_frame):
    velocity = np.asarray(velocity, dtype=np.float64)
    angular_velocity = np.asarray(angular_velocity, dtype=np.float64)
    if velocity.shape != (3,) or angular_velocity.shape != (3,):
        raise ValueError("own velocity and angular velocity must be three-vectors")
    if velocity_frame == 3:  # VehicleOdometry.VELOCITY_FRAME_BODY_FRD
        body_velocity = velocity
    elif velocity_frame in (1, 2) and pose_frame == velocity_frame:
        # PX4 q rotates body FRD into the odometry reference frame.
        body_velocity = quaternion_matrix(quaternion).T @ velocity
    else:
        raise ValueError("unknown or inconsistent PX4 velocity/orientation frames")
    result = np.concatenate((body_velocity, angular_velocity)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite PX4 own-vehicle state")
    return result


def past_self_state(history, image_timestamp_ns, maximum_age_seconds=.2):
    """Causal sample selection: no future sensor values or truth substitution."""
    for timestamp, values in reversed(history):
        age = (image_timestamp_ns-timestamp)*1e-9
        if age >= 0:
            return (timestamp, values) if age <= maximum_age_seconds else None
    return None
