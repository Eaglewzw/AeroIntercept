"""Camera geometry for training labels only; not an Actor observation."""
import numpy as np

CAMERA_LABEL_PROTOCOL = "gazebo_center_projection_letterbox_xy_v1"


def camera_target_xy(camera_relative_frd, horizontal_fov):
    """Project into [-1,1] coordinates of a square full-frame letterbox.

    Square pixels and a landscape source mean horizontal and vertical
    normalized coordinates share the same focal length. The validity mask
    comes from the separate physical visibility test; behind-camera samples
    carry finite placeholders but never supervise attention.
    """
    relative = np.asarray(camera_relative_frd, dtype=np.float64)
    if relative.shape != (3,) or not np.isfinite(relative).all():
        raise ValueError("camera projection requires a finite FRD three-vector")
    if not 0 < horizontal_fov < np.pi:
        raise ValueError("horizontal field of view must be in (0,pi)")
    if relative[0] <= 0:
        return np.zeros(2, dtype=np.float32)
    xy = relative[1:]/relative[0]/np.tan(horizontal_fov/2)
    return np.clip(xy, -10., 10.).astype(np.float32)
