"""Train a YOLO detector using the selected dataset's own class definitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.config import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/datasets/work_at_height/data.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="GPU index, or cpu")
    parser.add_argument("--project", type=Path, default=ROOT / "outputs/yolo")
    parser.add_argument("--name", default="yolo11s_rebuild")
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {args.data}")
    try:
        from ultralytics import YOLO
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install detector dependencies: python -m pip install -r requirements-yolo.txt") from exc
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("train") or not config.get("val"):
        raise ValueError("Dataset YAML must define train and val split paths")
    dataset_root = Path(config.get("path") or ".").expanduser()
    if not dataset_root.is_absolute():
        dataset_root = args.data.resolve().parent / dataset_root
    config["path"] = str(dataset_root.resolve())
    project = args.project.resolve()
    run_name = args.name
    if not run_name.strip() or Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("--name must be a directory name")
    run_dir = project / run_name
    suffix = 2
    while run_dir.exists():
        run_name = f"{args.name}{suffix}"
        run_dir = project / run_name
        suffix += 1
    run_dir.mkdir(parents=True)
    resolved_config = run_dir / "dataset_resolved.yaml"
    resolved_config.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    model = YOLO(args.model)
    model.train(data=str(resolved_config), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project=str(project),
                name=run_name, exist_ok=True)
    print(f"Resolved dataset config: {resolved_config}")
    print("Training finished. Check weights under the requested project/name directory.")


if __name__ == "__main__":
    main()
