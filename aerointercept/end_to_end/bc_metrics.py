"""Offline diagnostics in the same physical action units used by PX4."""
import numpy as np


class ActionDiagnostics:
    def __init__(self, velocity_max, yaw_rate_max):
        self.scales = np.asarray([velocity_max]*3+[yaw_rate_max], dtype=np.float64)
        self.groups = {}

    def update(self, prediction, target, mask, distances=None):
        def decode(values):
            values = np.clip(np.asarray(values, dtype=np.float64), -1., 1.).copy()
            values[..., :3] /= np.maximum(1., np.linalg.norm(values[..., :3], axis=-1, keepdims=True))
            return values*self.scales
        error = (decode(prediction)-decode(target)).reshape(-1, 4)
        valid = np.asarray(mask).reshape(-1) > 0
        groups = {"all": valid}
        if distances is not None:
            distance = np.asarray(distances).reshape(-1)
            groups.update({
                "below_1m": valid & (distance < 1.),
                "1_to_2m": valid & (distance >= 1.) & (distance < 2.),
                "2_to_5m": valid & (distance >= 2.) & (distance < 5.),
                "5m_and_above": valid & (distance >= 5.),
            })
        for name, selected in groups.items():
            values = error[selected]
            count, total, squared = self.groups.get(name, (0, np.zeros(4), np.zeros(4)))
            self.groups[name] = (count+len(values), total+values.sum(0), squared+(values**2).sum(0))

    def report(self):
        return {
            "axes": ["forward", "right", "down", "yaw_rate"],
            "units": ["m/s", "m/s", "m/s", "rad/s"],
            "groups": {name: {
                "frames": count,
                "rmse": (np.sqrt(squared/count).tolist() if count else None),
                "bias": (total/count).tolist() if count else None,
            } for name, (count, total, squared) in self.groups.items()},
        }
