"""Codex 起草，待队员核验。只读数据审计；不拟合、不求解、不读取视频像素。

AI-assisted: Codex (OpenAI, GPT-6 family; exact release date unavailable),
2026-11-20. Human verification pending; not a competition solver.
Run with the existing mathorcup Python; JSON is written only to stdout.
"""
from pathlib import Path
from collections import Counter
from decimal import Decimal
import argparse
import csv
import hashlib
import json
import platform
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def time_summary(values):
    a = np.asarray(values, dtype=float)
    dt = np.diff(a)
    return {"n": len(a), "first": float(a[0]), "last": float(a[-1]),
            "duplicates": int(len(a) - len(np.unique(a))),
            "nonincreasing_steps": int(np.sum(dt <= 0)),
            "dt_min_median_max": [float(np.min(dt)), float(np.median(dt)), float(np.max(dt))]}


def main(root):
    base = root / "00_赛题与规则/正式发布/C_Problem_Data/Restocking_Problem_and_Data_EN"
    task = base / "Restocking_Tasks_and_Constraints"
    split = rows(task / "record_split.csv")
    units = rows(task / "source_segment_index.csv")
    jobs = rows(task / "tasks.csv")
    families = rows(task / "job_families.csv")
    zones = rows(task / "service_zones.csv")
    conditions = json.loads((task / "common_conditions.json").read_text())
    by_uuid = {r["episode_uuid"]: r for r in split}
    out = {"provenance": "Codex 起草，待队员核验；2026-11-20；仅结构/训练样例审计",
           "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
           "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(task.iterdir()) if p.is_file()},
           "conditions": conditions, "families": families, "zones": zones}
    out["split"] = {"episode_rows": len(split), "unique_uuid": len(by_uuid),
                    "episodes": dict(Counter(r["split"] for r in split)),
                    "by_dataset": {ds: dict(Counter(r["split"] for r in split if r["dataset"] == ds))
                                   for ds in ["EgoDemo", "EgoPro", "EgoStandard"]},
                    "environments": {s: len({r["environment_id"] for r in split if r["split"] == s})
                                     for s in ["train", "validation", "test"]}}
    out["overlap_counts"] = {k: {a + "|" + b: len({r[k] for r in split if r["split"] == a} &
                                               {r[k] for r in split if r["split"] == b})
                                   for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
                             for k in ["environment_id", "scene_id", "operator_id"]}
    out["source"] = {"rows": len(units), "eligible": dict(Counter(r["eligible"] for r in units)),
                     "literal_float_threshold_disagreements": [r["unit_id"] for r in units
                         if (float(r["duration_s"]) >= 1.2) != (r["eligible"] == "1")],
                     "decimal_endpoint_threshold_disagreements": [r["unit_id"] for r in units
                         if (Decimal(r["end_s"]) - Decimal(r["start_s"]) >= Decimal("1.2")) != (r["eligible"] == "1")],
                     "duration_subtraction_max_abs_error": max(abs(float(r["duration_s"]) -
                         (float(r["end_s"]) - float(r["start_s"]))) for r in units),
                     "eligible_pools": {s: dict(Counter(r["family_id"] for r in units if r["split"] == s and r["eligible"] == "1"))
                                        for s in ["train", "validation", "test"]},
                     "train_family_environment_counts": {f: len({r["environment_id"] for r in units
                         if r["split"] == "train" and r["eligible"] == "1" and r["family_id"] == f}) for f in ["P", "A", "L"]},
                     "split_or_environment_mismatches": sum(any(r[k] != by_uuid[r["episode_uuid"]][k]
                         for k in ["dataset", "split", "environment_id"]) for r in units)}
    out["tasks"] = {"rows": len(jobs), "cases": dict(Counter(r["case_id"] for r in jobs)),
                    "predecessors": {r["job_id"]: r["predecessors"].split("|") for r in jobs if r["predecessors"]},
                    "weight_sums": {s: sum(float(r["priority_weight"]) for r in jobs if r["case_id"] == s)
                                    for s in ["S1", "S2", "S3"]}}
    names = ["record_index.csv", "missing_observation_experiments.md", "tasks_and_evaluation.md", "file_manifest.csv", "attachment_guide.md"]
    out["missing_files"] = {n: [str(p.relative_to(base)) for p in base.rglob(n)] for n in names}
    out["mcap_count"] = sum(p.is_file() and p.suffix.lower() == ".mcap" for p in base.rglob("*"))
    package = base.parent / "Restocking_Problem_and_Data_EN.zip"
    if package.exists():
        with zipfile.ZipFile(package) as z:
            out["inner_zip_inventory"] = {"entries": len(z.namelist()), "documentation_matches":
                [n for n in z.namelist() if Path(n).name in names or "Data_Documentation" in n]}
    out["inventories"], out["samples"], out["train_quality"] = {}, {}, {}
    seen = []
    session_uuids = {}
    for ds in ["EgoDemo", "EgoPro", "EgoStandard"]:
        folder = base / "Raw_Data" / ds
        sessions = sorted(p for p in folder.iterdir() if p.is_dir())
        files = [p for p in folder.rglob("*") if p.is_file()]
        count = 0
        disagreements = []
        quality_count = Counter()
        quality_columns = set()
        for session in sessions:
            if ds == "EgoDemo":
                meta = pd.read_parquet(session / "meta/episodes/chunk-000/file-000.parquet")
                uid = str(meta.episode_uuid.iloc[0])
                count += pq.ParquetFile(session / "data/chunk-000/file-000.parquet").metadata.num_rows
                actual = {k: str(meta[k].iloc[0]) for k in ["environment_id", "scene_id", "operator_id"]}
                if by_uuid[uid]["split"] == "train":
                    flags = json.loads((session / "bad_frame_ratio.json").read_text())
                    quality_count["frames"] += flags["total_frames"]
                    for r in flags["hand_bad_frame_details"]:
                        for side in ["left", "right"]:
                            if r.get(side + "_bad_reason"):
                                quality_count[side + ":" + str(r[side + "_bad_reason"])] += 1
            else:
                meta = {r["field"]: r["value"] for r in rows(session / "session_metadata.csv")}
                uid = meta["task_info.episode_uuid"]
                count += len(rows(session / "tables/pose_left_hand.csv"))
                actual = {k: meta[("operator." if k == "operator_id" else "task_info.") + k]
                          for k in ["environment_id", "scene_id", "operator_id"]}
                if by_uuid[uid]["split"] == "train":
                    quality = pd.read_csv(session / "tables/annotation_bad_frame_pose_hand.csv")
                    quality_count["frames"] += len(quality)
                    quality_count["is_bad_true"] += int(quality.is_bad.sum())
                    quality_columns.update(quality.columns)
                    for c in quality.columns:
                        if c.startswith("problem_type["):
                            quality_count.update({c + ":" + str(k): int(v) for k, v in quality[c].value_counts().items()})
            seen.append(uid)
            session_uuids[(ds, session.name)] = uid
            if any(actual[k] != by_uuid[uid][k] for k in actual):
                disagreements.append(session.name)
        out["inventories"][ds] = {"sessions": len(sessions), "files": len(files),
            "extensions": dict(Counter(p.suffix.lower() for p in files)), "trajectory_rows": count,
            "metadata_split_mismatches": disagreements,
            "video_session_counts": dict(Counter(sum(p.suffix == ".mp4" for p in s.rglob("*")) for s in sessions))}
        out["train_quality"][ds] = {"counts": dict(quality_count), "columns_union": sorted(quality_columns)}
        pick = next(r for r in units if r["dataset"] == ds and r["split"] == "train")
        session = folder / pick["sample"]
        result = {"path": str(session.relative_to(root)), "split": by_uuid[pick["episode_uuid"]]["split"],
                  "source_sample_uuid_matches": session_uuids[(ds, session.name)] == pick["episode_uuid"]}
        if ds == "EgoDemo":
            p = session / "data/chunk-000/file-000.parquet"
            frame = pd.read_parquet(p)
            info = json.loads((session / "meta/info.json").read_text())
            quality = json.loads((session / "bad_frame_ratio.json").read_text())
            ann = json.loads((session / "annotation.json").read_text())
            result.update({"numerical_file": str(p.relative_to(root)), "shape": list(frame.shape),
                "columns": list(frame.columns), "time": time_summary(frame.timestamp), "fps_metadata": info["fps"],
                "frame_index_contiguous": bool(np.array_equal(frame.frame_index, np.arange(len(frame)))),
                "hand_arrays": {}, "features": {k: v for k, v in info["features"].items() if "state" in k},
                "quality": {"total_frames": quality["total_frames"], "flagged_entries": len(quality["hand_bad_frame_details"]),
                    "left_flagged": sum(bool(r.get("left_bad_reason")) for r in quality["hand_bad_frame_details"]),
                    "right_flagged": sum(bool(r.get("right_bad_reason")) for r in quality["hand_bad_frame_details"]),
                    "all_indices_in_range": all(0 <= r["frame"] < len(frame) for r in quality["hand_bad_frame_details"])},
                "semantic_segments": len(ann["subtask_annotations"]),
                "semantic_keys": sorted(ann["subtask_annotations"][0])})
            for c in frame:
                if "hand_" in c:
                    a = np.stack([np.stack(v) for v in frame[c]])
                    result["hand_arrays"][c] = {"shape": list(a.shape), "nonfinite": int((~np.isfinite(a)).sum())}
        else:
            table = session / "tables"
            result["tables"] = {}
            all_times = []
            for name in ["pose_left_hand.csv", "pose_right_hand.csv", "pose_head.csv", "pose_headcam.csv", "pose_right_eye_cam.csv", "annotation_bad_frame_pose_hand.csv"]:
                f = pd.read_csv(table / name)
                tm = f["_offset_from_first_log_s"].to_numpy()
                all_times.append(tm)
                z = {"shape": list(f.shape), "time": time_summary(tm)}
                if name in ["pose_left_hand.csv", "pose_right_hand.csv"]:
                    xyz = [f"transforms[{k}].pos.{c}" for k in range(21) for c in "xyz"]
                    quat = [f"transforms[{k}].quat.{c}" for k in range(21) for c in "wxyz"]
                    z.update({"keypoint_counts": sorted(f["transforms.length"].unique().tolist()),
                        "position_columns": len(xyz), "quaternion_columns": len(quat),
                        "nonfinite": int((~np.isfinite(f[xyz + quat].to_numpy())).sum())})
                else:
                    z["columns"] = list(f.columns)
                if "bad_frame" in name:
                    z.update({"is_bad_true": int(f.is_bad.sum()), "reason_lengths": f["problem_type.length"].value_counts().to_dict()})
                result["tables"][name] = z
            result["timestamps_exactly_aligned"] = all(np.array_equal(a, all_times[0]) for a in all_times)
            semantic = pd.read_csv(table / "annotation_semantic_segments.csv")
            result["semantics"] = {"rows": len(semantic), "columns": list(semantic.columns)}
            ext = pd.read_csv(table / "sensor_camera_head_left_extrinsic.csv")
            result["left_extrinsic_frames"] = ext[["transforms[0].parent_frame_id", "transforms[0].child_frame_id"]].drop_duplicates().to_dict("records")
            result["video_frame_index_columns"] = list(pd.read_csv(session / "videos/head_left_frames.csv", nrows=1).columns)
            meta = rows(session / "session_metadata.csv")
            result["metadata_fields"] = [r["field"] for r in meta]
            result["trajectory_unit_fields_in_metadata"] = [r["field"] for r in meta
                if "unit" in r["field"].lower() and not r["field"].startswith("operator.")]
        out["samples"][ds] = result
    out["raw_uuid_check"] = {"records": len(seen), "unique": len(set(seen)),
                             "missing_from_split": len(set(seen) - set(by_uuid)),
                             "split_without_raw": len(set(by_uuid) - set(seen)),
                             "source_sample_uuid_mismatches": sum(session_uuids[(r["dataset"], r["sample"])] != r["episode_uuid"] for r in units)}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    main(parser.parse_args().root.resolve())
