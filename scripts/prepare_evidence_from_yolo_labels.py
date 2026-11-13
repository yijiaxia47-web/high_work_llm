"""Generate four-source inputs using dataset annotation boxes (oracle mode)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.config import ROOT
from main.evidence import CLASS_NAMES, collect_images, parse_yolo_label, write_evidence, write_manifest


def validate_dataset_classes(dataset_root: Path) -> None:
    config_path = dataset_root / "data.yaml"
    if not config_path.is_file():
        return  # Retain support for older datasets without a YAML config.
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = config.get("names") if isinstance(config, dict) else None
    if isinstance(names, dict):
        names = [names.get(index, names.get(str(index))) for index in range(len(names))]
    if names != CLASS_NAMES:
        raise ValueError(
            f"Dataset classes {names!r} do not match this six-class evidence pipeline: {CLASS_NAMES}. "
            "Keep independent datasets and class IDs separate; this dataset can use train_yolo.py or base mode."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/datasets/work_at_height")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--outdir", type=Path, default=ROOT / "data/evidence/oracle")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-classes", type=int, default=2)
    parser.add_argument("--allow-no-person", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0 or args.min_classes < 0:
        parser.error("--limit must be positive and --min-classes must be nonnegative")
    validate_dataset_classes(args.dataset_root)
    from PIL import Image

    image_dir = args.dataset_root / "images" / args.split
    label_dir = args.dataset_root / "labels" / args.split
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")
    entries = []
    for image_path in collect_images(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        objects = parse_yolo_label(label_path, *image.size)
        classes = {obj["class"] for obj in objects}
        if len(classes) < args.min_classes:
            continue
        if not args.allow_no_person and "person" not in classes:
            continue
        entries.append(write_evidence(image_path, image, objects, args.outdir,
                                      "ground_truth_boxes_oracle_detector"))
        if len(entries) >= args.limit:
            break
    if not entries:
        raise ValueError(f"No labeled images matched the filters in {image_dir}")
    manifest = write_manifest(entries, args.outdir)
    print(f"Generated {len(entries)} oracle samples: {manifest.resolve()}")
    print("Evidence mode: ground-truth boxes (oracle detector), not YOLO predictions")


if __name__ == "__main__":
    main()
