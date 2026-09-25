"""Full-frame, detector-free interception stack.

The actor consumes only a short history of RGB frames.  Simulator truth and
PNG detections are exposed exclusively as training labels or critic inputs.
"""

from .actions import (
    ACTION_DIM,
    BodyVelocityCommand,
    decode_action,
    encode_velocity_command,
)


def __getattr__(name):
    # System-Python ROS utilities need NumPy actions, not Gym or PyTorch.
    if name in ("EndToEndInterceptEnv", "VecEndToEndInterceptEnv"):
        from . import environment
        return getattr(environment, name)
    raise AttributeError(name)

__all__ = [
    "ACTION_DIM",
    "BodyVelocityCommand",
    "decode_action",
    "encode_velocity_command",
    "EndToEndInterceptEnv",
    "VecEndToEndInterceptEnv",
]
