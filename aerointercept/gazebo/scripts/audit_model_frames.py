"""Fail closed if resolved PX4 model frames differ from the task geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aerointercept.gazebo.frames import BODY_CENTER_OFFSET_FRD, CAMERA_OFFSET_FRD, PHYSICAL_STATE_SOURCE


def audit(models):
    import sdformat14
    report = {"physical_state_source": PHYSICAL_STATE_SOURCE, "models": {}}
    report["source_sdf_sha256"] = {
        name: hashlib.sha256((Path(models)/name/"model.sdf").read_bytes()).hexdigest()
        for name in ("x500_base", "OakD-Lite", "x500", "x500_depth")
    }
    for name in ("x500", "x500_depth"):
        root = sdformat14.Root()
        path = Path(models)/name/"model.sdf"
        errors = root.load(str(path))
        if errors:
            raise RuntimeError(f"cannot resolve {path}: {errors}")
        model = root.model()
        body = model.link_by_name("base_link").semantic_pose().resolve()
        values = lambda p: [p.x(), p.y(), p.z(), p.roll(), p.pitch(), p.yaw()]
        body_pose = values(body)
        expected = [BODY_CENTER_OFFSET_FRD[0], -BODY_CENTER_OFFSET_FRD[1], -BODY_CENTER_OFFSET_FRD[2], 0, 0, 0]
        if not np.allclose(body_pose, expected, atol=1e-9, rtol=0):
            raise RuntimeError(f"unexpected {name} body center transform: {body_pose}")
        item = {"base_link_in_model_flu": body_pose}
        if name == "x500_depth":
            camera = model.link_by_name("camera_link").sensor_by_name("IMX214")
            camera_pose = values(camera.semantic_pose().resolve("base_link"))
            expected = [CAMERA_OFFSET_FRD[0], -CAMERA_OFFSET_FRD[1], -CAMERA_OFFSET_FRD[2], 0, 0, 0]
            if not np.allclose(camera_pose, expected, atol=1e-9, rtol=0):
                raise RuntimeError(f"unexpected camera transform: {camera_pose}")
            item["camera_in_base_link_flu"] = camera_pose
        report["models"][name] = item
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(args.models)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print("verified resolved base_link centers and camera extrinsics", flush=True)


if __name__ == "__main__":
    main()
