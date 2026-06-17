import argparse
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.aggregate import aggregate_datasets
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
    """(H, W) → (H, W, 3): tile single-channel for the video encoder."""
    return np.repeat(arr[..., None], 3, axis=-1)


# ── Per-file worker (runs in a subprocess) ──────────────────────────────────────

def _process_file(h5_path: Path, temp_dir: Path) -> Path | None:
    task = _task_instruction(h5_path.name)
    if task is None:
        print(f"[skip] could not parse task from {h5_path.name}")
        return None

    temp_output = temp_dir / h5_path.stem
    if temp_output.exists():
        shutil.rmtree(temp_output)

    dataset = LeRobotDataset.create(
        repo_id=f"{h5_path.parent.name}/{h5_path.stem}",
        root=temp_output,
        fps=20,
        robot_type="franka",
        features=FEATURES,
    )

    with h5py.File(h5_path, "r") as f:
        for demo in f["data"].values():
            demo_len = len(demo["obs/agentview_rgb"])

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
            ee_pos            = np.array(demo["obs/ee_pos"],         dtype=np.float32)
            ee_ori            = np.array(demo["obs/ee_ori"],         dtype=np.float32)
            ee_state          = np.array(demo["obs/ee_states"],      dtype=np.float32)
            joint_state       = np.array(demo["obs/joint_states"],   dtype=np.float32)
            gripper_state     = np.array(demo["obs/gripper_states"], dtype=np.float32)
            robot_state       = np.array(demo["robot_states"],       dtype=np.float32)
            rewards           = np.array(demo["rewards"],            dtype=np.float32)[:, None]
            dones             = np.array(demo["dones"],              dtype=bool)[:, None]

            for i in range(demo_len):
                dataset.add_frame(
                    {
                        "observation.images.agentview_rgb":     agentview_rgb[i],
                        "observation.images.eye_in_hand_rgb":   eye_in_hand_rgb[i],
                        "observation.images.agentview_depth":   agentview_depth[i],
                        "observation.images.eye_in_hand_depth": eye_in_hand_depth[i],
                        "observation.images.agentview_seg":     agentview_seg[i],
                        "observation.images.eye_in_hand_seg":   eye_in_hand_seg[i],
                        "observation.state":                    state[i],
                        "observation.states.ee_pos":            ee_pos[i],
                        "observation.states.ee_ori":            ee_ori[i],
                        "observation.states.ee_state":          ee_state[i],
                        "observation.states.joint_state":       joint_state[i],
                        "observation.states.gripper_state":     gripper_state[i],
                        "observation.states.robot_state":       robot_state[i],
                        "action":                               action[i],
                        "next.reward":                          rewards[i],
                        "next.done":                            dones[i],
                    },
                    task,
                )
            dataset.save_episode()

    print(f"[done] {h5_path.name}")
    return temp_output


# ── Main ────────────────────────────────────────────────────────────────────────

def main(
    src_paths: list[Path],
    output_path: Path,
    repo_id: str,
    workers: int = -1,
    push_to_hub: bool = False,
):
    output_path = output_path.resolve()
    script_dir = Path(__file__).resolve().parent

    try:
        output_path.relative_to(script_dir)
        in_script_dir = True
    except ValueError:
        in_script_dir = False

    if in_script_dir or output_path == script_dir:
        raise ValueError(
            f"--output-path {output_path} is inside or equal to the script directory "
            f"{script_dir}. Choose a path outside the repo."
        )

    temp_dir = output_path.with_name(output_path.name + "_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    h5_files = [
        p for p in _iter_h5_files(src_paths)
        if _task_instruction(p.name) is not None
    ]
    if not h5_files:
        raise ValueError("No matching .hdf5 files found in --src-paths.")

    n_workers = os.cpu_count() or 1 if workers == -1 else workers
    print(f"Converting {len(h5_files)} files with {n_workers} workers ...")

    temp_outputs: list[Path] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_file, h5, temp_dir): h5
            for h5 in h5_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="files"):
            h5 = futures[future]
            try:
                result = future.result()
                if result is not None:
                    temp_outputs.append(result)
            except Exception as exc:
                print(f"[error] {h5.name}: {exc}")

    if not temp_outputs:
        raise RuntimeError("All workers failed — no output to aggregate.")

    print(f"Aggregating {len(temp_outputs)} temp datasets → {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)
    aggregate_datasets(
        repo_ids=[None] * len(temp_outputs),
        roots=temp_outputs,
        aggr_repo_id=repo_id,
        aggr_root=output_path,
    )

    shutil.rmtree(temp_dir)
    print(f"Done. Dataset written to {output_path}")

    if push_to_hub:
        LeRobotDataset(repo_id=repo_id, root=output_path).push_to_hub(
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
                        help="HuggingFace repo id, e.g. htrbao/LIBERO-MEM")
    parser.add_argument("--workers", type=int, default=-1,
                        help="Number of parallel workers (default: number of CPUs)")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    main(
        src_paths=args.src_paths,
        output_path=args.output_path,
        repo_id=args.repo_id,
        workers=args.workers,
        push_to_hub=args.push_to_hub,
    )
