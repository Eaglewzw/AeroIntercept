"""Gazebo Harmonic / PX4 end-to-end interception backend.

The CUDA training process deliberately does not import ROS 2.  ROS Humble is
built for the system Python while the pinned AeroIntercept Conda environment
uses Python 3.12, so :mod:`aerointercept.gazebo.ros_bridge` runs as a small
system-Python process and exchanges images and state over a local Unix socket.
"""

def __getattr__(name):
    if name in ("GazeboInterceptEnv", "GazeboVectorEnv"):
        from . import environment
        return getattr(environment, name)
    raise AttributeError(name)

__all__ = ["GazeboInterceptEnv", "GazeboVectorEnv"]
