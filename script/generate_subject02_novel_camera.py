from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def main():
    model_dir = Path("/home/hanlong/project/mmlphuman/output/subject02")
    out_dir = Path("/home/hanlong/project/mmlphuman/render/subject02_novelviewpose")
    out_dir.mkdir(parents=True, exist_ok=True)

    with (model_dir / "cameras.json").open("r") as f:
        cams = json.load(f)

    cam_a = np.array(cams[0]["w2c"], dtype=np.float32)
    cam_b = np.array(cams[1]["w2c"], dtype=np.float32)
    c2w_a = np.linalg.inv(cam_a)
    c2w_b = np.linalg.inv(cam_b)

    key_times = [0.0, 1.0]
    rots = Rotation.from_matrix(np.stack([c2w_a[:3, :3], c2w_b[:3, :3]], axis=0))
    interp_rot = Slerp(key_times, rots)([0.5]).as_matrix()[0]
    interp_t = 0.5 * (c2w_a[:3, 3] + c2w_b[:3, 3])

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = interp_rot
    c2w[:3, 3] = interp_t
    w2c = np.linalg.inv(c2w)

    K = 0.5 * (
        np.array(cams[0]["K"], dtype=np.float32) + np.array(cams[1]["K"], dtype=np.float32)
    )
    width = 1330
    height = 1150
    fovx = float(2 * np.arctan(width / (2 * K[0, 0])) * 180 / np.pi)

    cam_info = {
        "name": "subject02_mid_cam_00_01",
        "source_cam_ids": [int(cams[0]["cam_id"]), int(cams[1]["cam_id"])],
        "interp_alpha": 0.5,
        "w2c": w2c.tolist(),
        "fovx": fovx,
        "height": height,
        "width": width,
    }

    out_path = out_dir / "cam_novel_mid_00_01.json"
    out_path.write_text(json.dumps(cam_info, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
