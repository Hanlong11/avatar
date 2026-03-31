from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# SMPL-X body pose joint order after global orient.
BODY_JOINTS = {
    "left_hip": 0,
    "right_hip": 1,
    "spine1": 2,
    "left_knee": 3,
    "right_knee": 4,
    "spine2": 5,
    "left_ankle": 6,
    "right_ankle": 7,
    "spine3": 8,
    "left_foot": 9,
    "right_foot": 10,
    "neck": 11,
    "left_collar": 12,
    "right_collar": 13,
    "head": 14,
    "left_shoulder": 15,
    "right_shoulder": 16,
    "left_elbow": 17,
    "right_elbow": 18,
    "left_wrist": 19,
    "right_wrist": 20,
}


def body_slice(joint_name: str, axis: int | None = None):
    base = 3 + BODY_JOINTS[joint_name] * 3
    if axis is None:
        return slice(base, base + 3)
    return base + axis


def build_base_sequence(start: int = 2500, length: int = 240):
    src_path = Path("/home/hanlong/project/mmlphuman/subject02/smpl_params.npz")
    data = np.load(src_path)
    stop = start + length

    global_orient = data["global_orient"][start:stop].astype(np.float32)
    body_pose = data["body_pose"][start:stop].astype(np.float32)
    transl = data["transl"][start:stop].astype(np.float32)
    left_hand = data["left_hand_pose"][start:stop].astype(np.float32)
    right_hand = data["right_hand_pose"][start:stop].astype(np.float32)

    poses = np.concatenate(
        [
            global_orient,
            body_pose,
            np.zeros((length, 3), dtype=np.float32),
            np.zeros((length, 6), dtype=np.float32),
            left_hand,
            right_hand,
        ],
        axis=1,
    ).astype(np.float32)
    return poses, transl


def save_motion(out_dir: Path, name: str, poses: np.ndarray, trans: np.ndarray, description: str):
    np.savez_compressed(out_dir / f"{name}.npz", poses=poses.astype(np.float32), trans=trans.astype(np.float32))
    return {"file": f"{name}.npz", "frames": int(len(poses)), "description": description}


def make_backbend_twist(base_poses: np.ndarray, base_trans: np.ndarray):
    poses = base_poses.copy()
    trans = base_trans.copy()
    n = len(poses)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    wave = np.sin(2 * np.pi * t).astype(np.float32)
    wave2 = np.sin(4 * np.pi * t + 0.5).astype(np.float32)

    for joint, val in [("spine1", -0.35), ("spine2", -0.50), ("spine3", -0.60), ("neck", -0.18), ("head", -0.12)]:
        poses[:, body_slice(joint, 0)] += val + 0.08 * wave

    poses[:, 1] += 0.45 * wave  # global yaw-like twist
    poses[:, body_slice("spine2", 1)] += 0.35 * wave
    poses[:, body_slice("spine3", 1)] += 0.45 * wave

    poses[:, body_slice("left_shoulder", 2)] += 1.10 + 0.20 * wave2
    poses[:, body_slice("right_shoulder", 2)] -= 1.10 + 0.20 * wave2
    poses[:, body_slice("left_elbow", 1)] -= 0.85 + 0.15 * wave
    poses[:, body_slice("right_elbow", 1)] += 0.85 + 0.15 * wave

    poses[:, body_slice("left_hip", 0)] += 0.18
    poses[:, body_slice("right_hip", 0)] += 0.18
    poses[:, body_slice("left_knee", 0)] -= 0.22
    poses[:, body_slice("right_knee", 0)] -= 0.22
    trans[:, 2] += 0.02 * np.sin(2 * np.pi * t)
    return poses, trans


def make_high_kick_balance(base_poses: np.ndarray, base_trans: np.ndarray):
    poses = base_poses.copy()
    trans = base_trans.copy()
    n = len(poses)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    phase = np.sin(2 * np.pi * t).astype(np.float32)
    left_kick = np.clip(phase, 0.0, None)
    right_kick = np.clip(-phase, 0.0, None)

    poses[:, body_slice("left_hip", 0)] -= 1.10 * left_kick
    poses[:, body_slice("right_hip", 0)] -= 1.10 * right_kick
    poses[:, body_slice("left_knee", 0)] += 0.55 * left_kick
    poses[:, body_slice("right_knee", 0)] += 0.55 * right_kick
    poses[:, body_slice("left_ankle", 0)] -= 0.25 * left_kick
    poses[:, body_slice("right_ankle", 0)] -= 0.25 * right_kick

    poses[:, body_slice("left_hip", 2)] += 0.35 * left_kick
    poses[:, body_slice("right_hip", 2)] -= 0.35 * right_kick
    poses[:, body_slice("spine1", 2)] -= 0.18 * phase
    poses[:, body_slice("spine2", 2)] -= 0.22 * phase

    poses[:, body_slice("left_shoulder", 2)] += 0.95 - 0.25 * left_kick + 0.15 * right_kick
    poses[:, body_slice("right_shoulder", 2)] -= 0.95 - 0.25 * right_kick + 0.15 * left_kick
    poses[:, body_slice("left_elbow", 1)] -= 0.55
    poses[:, body_slice("right_elbow", 1)] += 0.55

    trans[:, 0] += 0.08 * phase
    trans[:, 2] += 0.05 * (left_kick + right_kick)
    return poses, trans


def make_spin_punch_combo(base_poses: np.ndarray, base_trans: np.ndarray):
    poses = base_poses.copy()
    trans = base_trans.copy()
    n = len(poses)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    spin = 2.0 * np.pi * t
    jab = np.sin(6 * np.pi * t).astype(np.float32)

    poses[:, 1] += spin.astype(np.float32)  # full spin in global orientation
    poses[:, body_slice("spine1", 1)] += 0.25 * np.sin(2 * np.pi * t)
    poses[:, body_slice("spine2", 1)] += 0.35 * np.sin(2 * np.pi * t)
    poses[:, body_slice("spine3", 1)] += 0.45 * np.sin(2 * np.pi * t)

    poses[:, body_slice("left_shoulder", 2)] += 0.65
    poses[:, body_slice("right_shoulder", 2)] -= 0.65
    poses[:, body_slice("left_elbow", 1)] -= 0.35
    poses[:, body_slice("right_elbow", 1)] += 0.35

    poses[:, body_slice("left_shoulder", 0)] += 0.55 * np.clip(jab, 0.0, None)
    poses[:, body_slice("left_elbow", 0)] -= 0.45 * np.clip(jab, 0.0, None)
    poses[:, body_slice("right_shoulder", 0)] += 0.55 * np.clip(-jab, 0.0, None)
    poses[:, body_slice("right_elbow", 0)] -= 0.45 * np.clip(-jab, 0.0, None)

    poses[:, body_slice("left_hip", 2)] += 0.12 * np.sin(4 * np.pi * t)
    poses[:, body_slice("right_hip", 2)] -= 0.12 * np.sin(4 * np.pi * t)
    poses[:, body_slice("left_knee", 0)] -= 0.18
    poses[:, body_slice("right_knee", 0)] -= 0.18

    trans[:, 0] += 0.06 * np.sin(2 * np.pi * t)
    trans[:, 1] += 0.04 * np.cos(2 * np.pi * t)
    return poses, trans


def main():
    out_dir = Path("/home/hanlong/project/mmlphuman/custom_motions")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_poses, base_trans = build_base_sequence()
    manifest = []

    motions = {
        "extreme_backbend_twist": (
            make_backbend_twist,
            "Strong backbend with torso twist and overhead arm sweep.",
        ),
        "high_kick_balance": (
            make_high_kick_balance,
            "Alternating high kicks with side balance and counter-arm motion.",
        ),
        "spin_punch_combo": (
            make_spin_punch_combo,
            "Fast global spin with alternating jab-like upper-body motion.",
        ),
    }

    for name, (builder, description) in motions.items():
        poses, trans = builder(base_poses, base_trans)
        manifest.append(save_motion(out_dir, name, poses, trans, description))

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(out_dir)
    for item in manifest:
        print(f"{item['file']}: {item['description']}")


if __name__ == "__main__":
    main()
