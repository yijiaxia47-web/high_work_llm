"""Copy the separate four-class harness dataset without changing originals.

Default: preview. --apply copies and verifies each destination against the
source SHA-256. --verify checks the persisted inventory, even after source removal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.migrate_experiments import Plan, checked_path, digest, json_bytes, resolved_directory

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("input_demo/originals/高空作业安全带佩戴检测")
DESTINATION = Path("data/datasets/harness_wearing")
REPORT = Path("outputs/migration_harness_wearing/report.json")
INVENTORY = REPORT.with_name("files.csv")
SPLITS = ("train", "val", "test")


def inspect_dataset(folder: Path) -> dict:
    names = (folder / "labels/labels.txt").read_text(encoding="utf-8-sig").splitlines()
    if not names or any(not name.strip() for name in names) or len(names) != len(set(names)):
        raise ValueError("Invalid or duplicate class names in labels/labels.txt")
    counts = {}
    objects = Counter()
    for split in SPLITS:
        images = [p for p in (folder / "images" / split).iterdir() if p.is_file()]
        labels = list((folder / "labels" / split).glob("*.txt"))
        image_ids, label_ids = {p.stem for p in images}, {p.stem for p in labels}
        if not images or len(image_ids) != len(images) or image_ids != label_ids:
            raise ValueError(f"Images and labels are not paired uniquely in {split}")
        for label in labels:
            for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Expected YOLO detection label: {label}:{line_number}")
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
                if not 0 <= class_id < len(names) or not all(math.isfinite(v) and 0 <= v <= 1 for v in coordinates):
                    raise ValueError(f"Invalid YOLO class or coordinates: {label}:{line_number}")
                objects[class_id] += 1
        counts[split] = len(images)
    return {"names": names, "split_pairs": counts,
            "objects_by_class_id": {str(k): objects[k] for k in range(len(names))}}


def dataset_yaml(names: list[str]) -> bytes:
    lines = ["# Independent harness-wearing dataset; retain original class IDs and val split.",
             "path: .", "train: images/train", "val: images/val", "test: images/test",
             f"nc: {len(names)}", "names:"]
    lines.extend(f"  - {json.dumps(name, ensure_ascii=False)}" for name in names)
    return ("\n".join(lines) + "\n").encode("utf-8")


def source_files(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in (root / SOURCE).rglob("*") if p.is_file()}


def verify(root: Path) -> None:
    resolved_directory.cache_clear()
    report = json.loads((root / REPORT).read_text(encoding="utf-8"))
    if digest(root / INVENTORY) != report["inventory_sha256"]:
        raise ValueError("Migration inventory hash mismatch")
    with (root / INVENTORY).open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    expected_sources = set()
    for index, row in enumerate(records, 1):
        target = checked_path(root, Path(row["destination"]))
        if not target.is_file() or target.stat().st_size != int(row["bytes"]) or digest(target) != row["sha256"]:
            raise ValueError(f"Missing or changed migrated file: {target}")
        if row["source"]:
            relative = Path(row["source"])
            expected_sources.add(relative)
            source = checked_path(root, relative)
            if source.exists() and digest(source) != row["sha256"]:
                raise ValueError(f"Original changed since migration: {source}")
        if index % 500 == 0:
            print(f"Verified {index}/{len(records)} files", flush=True)
    if (root / SOURCE).exists() and source_files(root) != expected_sources:
        raise ValueError("Original file inventory changed")
    if inspect_dataset(root / DESTINATION) != report["dataset"]:
        raise ValueError("Migrated class definitions or split labels changed")
    print(f"Verified {len(records)} files; independent dataset preserved.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(ROOT)
        return
    print("Inspecting original classes and image-label pairs...", flush=True)
    metadata = inspect_dataset(ROOT / SOURCE)
    originals = source_files(ROOT)
    if not originals:
        raise ValueError("No source files found")
    plan = Plan(ROOT)
    for index, source in enumerate(sorted(originals), 1):
        plan.copy(source, DESTINATION / source.relative_to(SOURCE))
        if index % 500 == 0:
            print(f"Preflight: {index}/{len(originals)} source files", flush=True)
    plan.add(DESTINATION / "data.yaml", dataset_yaml(metadata["names"]))
    print(f"Plan: {len(originals)} original files, {sum(v['bytes'] for v in plan.entries.values()):,} bytes; no source deletion.", flush=True)
    if not args.apply:
        print("Preview only. Use --apply to copy and verify.")
        return
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["source", "destination", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(plan.entries.values())
    inventory = output.getvalue().encode("utf-8")
    report = {
        "source": SOURCE.as_posix(), "destination": DESTINATION.as_posix(),
        "source_files": len(originals), "source_bytes": sum(v["bytes"] for v in plan.entries.values() if v["source"]),
        "dataset": metadata, "inventory": INVENTORY.as_posix(),
        "inventory_sha256": hashlib.sha256(inventory).hexdigest(),
        "verification": "Each destination SHA-256 matched source SHA-256; original files retained.",
        "policy": "Separate dataset; preserve filenames, train/val/test and all class IDs; no overwrite or deletion.",
    }
    plan.add(INVENTORY, inventory)
    plan.add(REPORT, json_bytes(report))
    plan.apply(progress_every=500)
    if source_files(ROOT) != originals or inspect_dataset(ROOT / DESTINATION) != metadata:
        raise ValueError("Source inventory or migrated dataset validation failed")
    print(f"Copied and SHA-256 verified {len(originals)} originals plus independent data.yaml.")
    print(f"Report: {REPORT.as_posix()}")


if __name__ == "__main__":
    main()
