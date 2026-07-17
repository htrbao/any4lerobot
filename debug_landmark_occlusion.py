"""
Diagnose why a landmark (e.g. 'cream_cheese') isn't showing amodal inpainting
in liberomem2lerobot/liberomem_w_blur_h5.py.

For a given .hdf5 + its sibling metainfo.json, prints:
  - every landmark name seen in exo_boxes/ego_boxes for one demo
  - how many frames each name is present/missing in the metainfo dict
  - the seg_id metainfo recorded for that name
  - whether that seg_id actually shows up in obs/agentview_seg (or eye_in_hand_seg)

Usage:
    python debug_landmark_occlusion.py /path/to/some_demo.hdf5
    python debug_landmark_occlusion.py /path/to/some_demo.hdf5 --demo demo_3 --name cheese
    python debug_landmark_occlusion.py /path/to/some_demo.hdf5 --boxes-key ego_boxes
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np


def _metadata_task_key(h5_stem: str, metadata: dict) -> str | None:
    if h5_stem in metadata:
        return h5_stem
    stripped = re.sub(r"_demo$", "", h5_stem)
    if stripped in metadata:
        return stripped
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5_path", type=Path)
    ap.add_argument("--demo", default=None, help="demo key, e.g. demo_0 (default: first)")
    ap.add_argument("--name", default=None, help="landmark name substring filter, e.g. cheese")
    ap.add_argument("--boxes-key", default="exo_boxes", choices=["exo_boxes", "ego_boxes"])
    ap.add_argument("--seg-key", default=None,
                     help="obs/<seg-key> to check seg_id against "
                          "(default: obs/agentview_seg for exo_boxes, obs/eye_in_hand_seg for ego_boxes)")
    args = ap.parse_args()

    seg_key = args.seg_key or ("agentview_seg" if args.boxes_key == "exo_boxes" else "eye_in_hand_seg")

    meta_path = args.hdf5_path.parent / "metainfo.json"
    if not meta_path.exists():
        print(f"[!] no metainfo.json next to {args.hdf5_path}")
        return
    metadata = json.loads(meta_path.read_text())

    task_key = _metadata_task_key(args.hdf5_path.stem, metadata)
    if task_key is None:
        print(f"[!] no metainfo.json entry matches h5 stem '{args.hdf5_path.stem}'. "
              f"Available keys (first 5): {list(metadata)[:5]}")
        return
    task_entry = metadata[task_key]

    with h5py.File(args.hdf5_path, "r") as f:
        demo_names = list(f["data"].keys())
        demo_name = args.demo or demo_names[0]
        demo = f["data"][demo_name]
        demo_len = len(demo["obs/agentview_rgb"])
        seg_video = np.array(demo[f"obs/{seg_key}"])
        print(f"demo={demo_name}  demo_len={demo_len}  {seg_key} shape={seg_video.shape} dtype={seg_video.dtype}")
        print(f"unique values across the whole episode's {seg_key}: {np.unique(seg_video)}")

        demo_meta = task_entry.get(demo_name, {})
        if not demo_meta:
            print(f"[!] no metainfo entry for demo '{demo_name}' under task '{task_key}'. "
                  f"Available demo keys (first 5): {list(task_entry)[:5]}")
            return
        boxes_list = demo_meta.get(args.boxes_key, [])
        print(f"{args.boxes_key}: {len(boxes_list)} frame-entries (demo_len={demo_len})")

        all_names = set()
        for fb in boxes_list:
            all_names.update(fb.keys())
        print(f"all landmark names seen in {args.boxes_key}: {sorted(all_names)}")

        target_names = (
            [n for n in all_names if args.name.lower() in n.lower()] if args.name else sorted(all_names)
        )
        if args.name and not target_names:
            print(f"[!] no landmark name matched substring '{args.name}'")

        for name in target_names:
            present_frames = [i for i, fb in enumerate(boxes_list) if name in fb]
            missing_frames = [i for i in range(len(boxes_list)) if name not in boxes_list[i]]
            seg_ids_seen = {boxes_list[i][name][0] for i in present_frames}

            print(f"\n--- landmark '{name}' ---")
            print(f"  present in metainfo dict for {len(present_frames)}/{len(boxes_list)} frames")
            print(f"  MISSING from metainfo dict for {len(missing_frames)}/{len(boxes_list)} frames"
                  + (f" e.g. {missing_frames[:10]}" if missing_frames else " (dict-presence alone would never flag this as occluded)"))
            print(f"  seg_id(s) recorded in metainfo across those frames: {seg_ids_seen}")

            if present_frames:
                sample_i = present_frames[0]
                seg_id, bbox, subgoal = boxes_list[sample_i][name]
                print(f"  sample: frame {sample_i} -> seg_id={seg_id}, bbox={bbox}, subgoal={subgoal!r}")

                sampled = present_frames[: min(30, len(present_frames))]
                match_count = sum(
                    1 for i in sampled
                    if i < len(seg_video) and np.any(seg_video[i] == boxes_list[i][name][0])
                )
                print(f"  seg_id found in obs/{seg_key} for {match_count}/{len(sampled)} sampled present-frames")
                if match_count == 0:
                    print(f"  -> seg_id from metainfo NEVER appears in obs/{seg_key}'s pixel values.\n"
                          f"     These two are on different numbering schemes -- the seg-based occlusion\n"
                          f"     check in liberomem_w_blur_h5.py can't work against this seg_key as-is.")
                elif match_count < len(sampled):
                    print(f"  -> seg_id sometimes matches, sometimes not: real visual occlusion IS being "
                          f"detected for at least some frames.")


if __name__ == "__main__":
    main()
