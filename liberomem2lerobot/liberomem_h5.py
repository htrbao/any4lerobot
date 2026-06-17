import argparse
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

FEATURES = {
    # ── RGB images ──────────────────────────────────────────────────────────────
    "observation.images.agentview_rgb": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.eye_in_hand_rgb": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    # ── Depth maps (tiled to 3 ch — LeRobot video encoder requires RGB) ─────────
    "observation.images.agentview_depth": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.eye_in_hand_depth": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    # ── Segmentation maps (tiled to 3 ch) ───────────────────────────────────────
    "observation.images.agentview_seg": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.eye_in_hand_seg": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    # ── Proprioception ───────────────────────────────────────────────────────────
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper"]},
    },
    "observation.states.ee_pos": {
        "dtype": "float32",
        "shape": (3,),
        "names": {"motors": ["x", "y", "z"]},
    },
    "observation.states.ee_ori": {
        "dtype": "float32",
        "shape": (3,),
        "names": {"motors": ["axis_angle1", "axis_angle2", "axis_angle3"]},
    },
    "observation.states.ee_state": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3"]},
    },
    "observation.states.joint_state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]},
    },
    "observation.states.gripper_state": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"motors": ["gripper", "gripper"]},
    },
    "observation.states.robot_state": {
        "dtype": "float32",
        "shape": (9,),
        "names": {"motors": ["motor_0", "motor_1", "motor_2", "motor_3", "motor_4", "motor_5", "motor_6", "motor_7", "motor_8"]},
    },
    # ── Action ───────────────────────────────────────────────────────────────────
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]},
    },
    # ── RL annotations ───────────────────────────────────────────────────────────
    "next.reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["reward"],
    },
    "next.done": {
        "dtype": "bool",
        "shape": (1,),
        "names": ["done"],
    },
}

_PATTERN1 = re.compile(r"_SCENE\d+_(.*?)_demo\.hdf5")
_PATTERN2 = re.compile(r"(.*?)_demo\.hdf5")


def _task_instruction(filename: str) -> str | None:
    match = _PATTERN1.search(filename) or _PATTERN2.search(filename)
    return match.group(1).replace("_", " ") if match else None


def _iter_h5_files(src_paths: list[Path]):
    for src in src_paths:
        if src.is_file() and src.suffix == ".hdf5":
            yield src
        elif src.is_dir():
            yield from sorted(src.glob("*.hdf5"))


def _expand(arr: np.ndarray) -> np.ndarray:
    """(H, W) → (H, W, 3): tile single-channel to RGB for the video encoder."""
    return np.repeat(arr[..., None], 3, axis=-1)


def main(
    src_paths: list[Path],
    output_path: Path,
    repo_id: str,
    push_to_hub: bool = False,
):
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        fps=20,
        robot_type="franka",
        features=FEATURES,
    )

    for h5_path in _iter_h5_files(src_paths):
        task = _task_instruction(h5_path.name)
        if task is None:
            print(f"[skip] could not parse task from {h5_path.name}")
            continue

        print(f"Processing {h5_path.name}  →  task: {task!r}")
        with h5py.File(h5_path, "r") as f:
            for demo in tqdm(f["data"].values(), desc=h5_path.stem, leave=False):
                demo_len = len(demo["obs/agentview_rgb"])

                # (-1: open, 1: close) → (0: close, 1: open)
                action = np.array(demo["actions"], dtype=np.float32)
                action = np.concatenate(
                    [action[:, :6], (1 - np.clip(action[:, -1:], 0, 1))],
                    axis=1,
                )

                state = np.concatenate(
                    [np.array(demo["obs/ee_states"]), np.array(demo["obs/gripper_states"])],
                    axis=1,
                ).astype(np.float32)

                agentview_rgb     = np.array(demo["obs/agentview_rgb"])
                eye_in_hand_rgb   = np.array(demo["obs/eye_in_hand_rgb"])
                agentview_depth   = _expand(np.array(demo["obs/agentview_depth"]))
                eye_in_hand_depth = _expand(np.array(demo["obs/eye_in_hand_depth"]))
                agentview_seg     = _expand(np.array(demo["obs/agentview_seg"]))
                eye_in_hand_seg   = _expand(np.array(demo["obs/eye_in_hand_seg"]))
                ee_pos            = np.array(demo["obs/ee_pos"],       dtype=np.float32)
                ee_ori            = np.array(demo["obs/ee_ori"],       dtype=np.float32)
                ee_state          = np.array(demo["obs/ee_states"],    dtype=np.float32)
                joint_state       = np.array(demo["obs/joint_states"], dtype=np.float32)
                gripper_state     = np.array(demo["obs/gripper_states"], dtype=np.float32)
                robot_state       = np.array(demo["robot_states"],     dtype=np.float32)
                rewards           = np.array(demo["rewards"],          dtype=np.float32)[:, None]
                dones             = np.array(demo["dones"],            dtype=bool)[:, None]

                for i in range(demo_len):
                    dataset.add_frame(
                        {
                            "observation.images.agentview_rgb":   agentview_rgb[i],
                            "observation.images.eye_in_hand_rgb": eye_in_hand_rgb[i],
                            "observation.images.agentview_depth":   agentview_depth[i],
                            "observation.images.eye_in_hand_depth": eye_in_hand_depth[i],
                            "observation.images.agentview_seg":     agentview_seg[i],
                            "observation.images.eye_in_hand_seg":   eye_in_hand_seg[i],
                            "observation.state":                  state[i],
                            "observation.states.ee_pos":          ee_pos[i],
                            "observation.states.ee_ori":          ee_ori[i],
                            "observation.states.ee_state":        ee_state[i],
                            "observation.states.joint_state":     joint_state[i],
                            "observation.states.gripper_state":   gripper_state[i],
                            "observation.states.robot_state":     robot_state[i],
                            "action":                             action[i],
                            "next.reward":                        rewards[i],
                            "next.done":                          dones[i],
                        },
                        task,
                    )
                dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero-mem", "franka"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-paths", type=Path, nargs="+", required=True,
                        help="One or more .hdf5 files or directories containing .hdf5 files")
    parser.add_argument("--output-path", type=Path, required=True,
                        help="Root directory for the output LeRobotDataset")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HuggingFace repo id, e.g. your_name/libero_mem")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    main(
        src_paths=args.src_paths,
        output_path=args.output_path,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
    )
