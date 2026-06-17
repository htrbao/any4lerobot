import argparse
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

EPISODES_PER_CHUNK = 1000  # must match LeRobot default

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

_PATTERN1 = re.compile(r"_SCENE\d_\d+_(.*?)_demo\.hdf5")
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


# ── Custom aggregation (avoids lerobot version-specific aggregate_datasets) ──────

def _merge_temp_datasets(temp_dirs: list[Path], output_dir: Path, repo_id: str) -> None:
    """Merge independently written temp LeRobot datasets into one output dataset.

    Handles tasks deduplication, global episode/frame re-indexing, parquet
    column patching, and video file copying without relying on lerobot internals.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    # Use first temp dataset's info.json as the base
    with open(temp_dirs[0] / "meta" / "info.json") as f:
        info = json.load(f)

    # Collect video feature keys (for file copying)
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]

    global_task_index: dict[str, int] = {}   # task_str → global idx
    next_task_idx = 0
    global_episodes: list[dict] = []
    global_ep_idx = 0
    global_frame_idx = 0

    for temp_dir in sorted(temp_dirs):
        # ── Map local task indices → global ──────────────────────────────────
        local_task_map: dict[int, int] = {}
        tasks_file = temp_dir / "meta" / "tasks.jsonl"
        if tasks_file.exists():
            for line in tasks_file.read_text().splitlines():
                if not line.strip():
                    continue
                t = json.loads(line)
                task_str = t["task"]
                local_idx = t["task_index"]
                if task_str not in global_task_index:
                    global_task_index[task_str] = next_task_idx
                    next_task_idx += 1
                local_task_map[local_idx] = global_task_index[task_str]

        # ── Process each episode in this temp dataset ─────────────────────────
        episodes_file = temp_dir / "meta" / "episodes.jsonl"
        if not episodes_file.exists():
            continue
        local_eps = [
            json.loads(l) for l in episodes_file.read_text().splitlines() if l.strip()
        ]

        for ep in local_eps:
            local_ep_idx: int = ep["episode_index"]
            ep_len: int = ep.get("length", 0)

            # Build updated episode record
            ep_record = dict(ep)
            ep_record["episode_index"] = global_ep_idx
            if "tasks" in ep_record:
                ep_record["tasks"] = [local_task_map.get(t, t) for t in ep_record["tasks"]]

            # ── Copy & patch parquet ──────────────────────────────────────────
            src_chunk = local_ep_idx // EPISODES_PER_CHUNK
            dst_chunk = global_ep_idx // EPISODES_PER_CHUNK
            src_parquet = (
                temp_dir / "data" / f"chunk-{src_chunk:03d}"
                / f"episode_{local_ep_idx:06d}.parquet"
            )
            dst_parquet_dir = output_dir / "data" / f"chunk-{dst_chunk:03d}"
            dst_parquet_dir.mkdir(parents=True, exist_ok=True)
            dst_parquet = dst_parquet_dir / f"episode_{global_ep_idx:06d}.parquet"

            if src_parquet.exists():
                table = pq.read_table(src_parquet)
                n = len(table)

                def _replace_col(tbl, col_name, values, dtype):
                    if col_name in tbl.schema.names:
                        idx = tbl.schema.get_field_index(col_name)
                        tbl = tbl.set_column(idx, col_name, pa.array(values, type=dtype))
                    return tbl

                table = _replace_col(table, "episode_index",
                                     [global_ep_idx] * n, pa.int64())
                table = _replace_col(table, "index",
                                     list(range(global_frame_idx, global_frame_idx + n)),
                                     pa.int64())
                if local_task_map:
                    if "task_index" in table.schema.names:
                        old = table.column("task_index").to_pylist()
                        table = _replace_col(table, "task_index",
                                             [local_task_map.get(t, t) for t in old],
                                             pa.int64())
                pq.write_table(table, dst_parquet)

            # ── Copy video files ──────────────────────────────────────────────
            for key in video_keys:
                src_vid_dir = (
                    temp_dir / "videos" / f"chunk-{src_chunk:03d}" / key
                )
                dst_vid_dir = output_dir / "videos" / f"chunk-{dst_chunk:03d}" / key
                dst_vid_dir.mkdir(parents=True, exist_ok=True)
                src_vid = src_vid_dir / f"episode_{local_ep_idx:06d}.mp4"
                if src_vid.exists():
                    shutil.copy2(src_vid, dst_vid_dir / f"episode_{global_ep_idx:06d}.mp4")

            global_episodes.append(ep_record)
            global_frame_idx += ep_len
            global_ep_idx += 1

    # ── Write combined metadata ───────────────────────────────────────────────────
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task_str, task_idx in sorted(global_task_index.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": task_idx, "task": task_str}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in global_episodes:
            f.write(json.dumps(ep) + "\n")

    info["repo_id"] = repo_id
    info["total_episodes"] = global_ep_idx
    info["total_frames"] = global_frame_idx
    info["total_tasks"] = len(global_task_index)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Merged {global_ep_idx} episodes · {global_frame_idx} frames · {len(global_task_index)} tasks → {output_dir}")


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
    _merge_temp_datasets(temp_outputs, output_path, repo_id)

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
