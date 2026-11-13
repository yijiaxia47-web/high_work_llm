"""Copy legacy artifacts safely; default to preview, never delete or overwrite.

Run --apply once, then --verify before removing experiments manually.
Historical archives retain their original bytes; working manifests are rebased.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.conditions import make_conditions
from main.manifest import load_manifest, validate_samples

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("outputs/migration_experiments/report.json")
INVENTORY = REPORT.with_name("files.csv")
DATASET = Path("input_demo/originals/Work at Height Safety Datase(DST2189)")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hash_stream(handle)


def hash_stream(handle) -> str:
    value = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        value.update(chunk)
    return value.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


@lru_cache(maxsize=32)
def resolved_root(root: Path) -> Path:
    return root.resolve()


@lru_cache(maxsize=512)
def resolved_directory(path: Path) -> Path:
    # Thousands of files share the same parent. Resolve junctions and symlinks
    # once per phase rather than repeating every ancestor lookup per file.
    return path.resolve()


def checked_path(root: Path, relative: Path) -> Path:
    base = resolved_root(root)
    candidate = base / relative
    path = resolved_directory(candidate.parent) / candidate.name
    if candidate.name in {".", ".."} or candidate.is_symlink():
        path = candidate.resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Path escapes project: {relative}")
    return path


class Plan:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.entries: dict[Path, dict] = {}
        self.payloads: dict[Path, bytes | Path] = {}
        self.gold_provenance: dict = {}

    def add(self, destination: Path, payload: bytes | Path, source: Path | None = None):
        target = checked_path(self.root, destination)
        if destination in self.entries:
            raise ValueError(f"Duplicate destination: {destination}")
        sha = digest(payload) if isinstance(payload, Path) else hashlib.sha256(payload).hexdigest()
        size = payload.stat().st_size if isinstance(payload, Path) else len(payload)
        # Preflight every file before the first write. Equal files are reusable.
        if target.exists() and (not target.is_file() or digest(target) != sha):
            raise FileExistsError(f"Conflicting destination; nothing overwritten: {target}")
        self.entries[destination] = {
            "source": source.as_posix() if source else "",
            "destination": destination.as_posix(), "bytes": size, "sha256": sha,
        }
        self.payloads[destination] = payload
        if len(self.entries) % 5000 == 0:
            print(f"Preflight: {len(self.entries)} files hashed and checked", flush=True)

    def copy(self, source: Path, destination: Path):
        self.add(destination, checked_path(self.root, source), source)

    def apply(self, progress_every: int = 5000):
        resolved_directory.cache_clear()
        for index, (destination, payload) in enumerate(self.payloads.items(), 1):
            target = checked_path(self.root, destination)
            if target.exists():
                if digest(target) != self.entries[destination]["sha256"]:
                    raise FileExistsError(f"Destination changed after preflight: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also protects against changes after preflight.
            with target.open("xb") as out:
                if isinstance(payload, Path):
                    with payload.open("rb") as source:
                        shutil.copyfileobj(source, out)
                else:
                    out.write(payload)
            if isinstance(payload, Path):
                shutil.copystat(payload, target)
            if digest(target) != self.entries[destination]["sha256"]:
                raise IOError(f"Copy verification failed: {target}")
            if index % progress_every == 0:
                print(f"Copied and verified: {index}/{len(self.entries)} files", flush=True)


def mapped_path(value: str) -> Path:
    normalized = value.replace("\\", "/")
    # Old records may have a different absolute project root. Match the known
    # directory suffix rather than assuming the current drive or root name.
    for prefix, destination in (
        (DATASET.as_posix() + "/", Path("data/datasets/work_at_height")),
        ("experiments/runs/oracle_evidence_002/", Path("data/evidence/oracle")),
        ("experiments/runs/oracle_evidence_001/", Path("data/evidence/oracle_001")),
        ("experiments/runs/gold_draft_001/gold/", Path("data/gold")),
        ("experiments/runs/", Path("outputs/legacy")),
    ):
        index = normalized.find(prefix)
        if index >= 0 and (index == 0 or normalized[index - 1] == "/"):
            return destination / normalized[index + len(prefix):]
    raise ValueError(f"Unrecognized legacy path; needs a mapping: {value}")


def rebase_manifest(value, destination: Path, root: Path):
    items = value if isinstance(value, list) else value["samples"]
    for item in items:
        for key in ("original_image", "numbered_visualization", "scene_graph", "rule_result", "gold"):
            if item.get(key):
                target = checked_path(root, mapped_path(item[key]))
                item[key] = Path(os.path.relpath(target, root / destination.parent)).as_posix()
    return value


def read_csv(path: Path):
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp936")
    return list(csv.DictReader(io.StringIO(text)))


def build_plan(root: Path) -> Plan:
    resolved_directory.cache_clear()
    plan = Plan(root)
    legacy = root / "experiments"
    if not legacy.is_dir():
        raise FileNotFoundError("experiments is absent; use --verify to check the completed migration")
    for source in sorted(p for p in legacy.rglob("*") if p.is_file()):
        relative = source.relative_to(legacy)
        destination = Path("outputs/legacy") / relative.relative_to("runs") if relative.parts[0] == "runs" else Path("outputs/legacy/scripts") / relative
        plan.copy(source.relative_to(root), destination)
    for category in ("images", "labels"):
        folder = root / DATASET / category
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for source in sorted(p for p in folder.rglob("*") if p.is_file()):
            plan.copy(source.relative_to(root), Path("data/datasets/work_at_height") / category / source.relative_to(folder))
    for name, destination in (("oracle_evidence_001", Path("data/evidence/oracle_001")), ("oracle_evidence_002", Path("data/evidence/oracle"))):
        folder = legacy / "runs" / name
        for source in sorted(p for p in folder.rglob("*") if p.is_file()):
            target = destination / source.relative_to(folder)
            if source.name.startswith("manifest") and source.suffix == ".json":
                value = json.loads(source.read_text(encoding="utf-8"))
                plan.add(target, json_bytes(rebase_manifest(value, target, root)))
            else:
                plan.copy(source.relative_to(root), target)

    drafts = legacy / "runs/gold_draft_001"
    # Only promote gold when the checked review reproduces both saved versions.
    with tempfile.TemporaryDirectory(prefix="verify_legacy_gold_") as temporary:
        subprocess.run([
            sys.executable, str(root / "scripts/finalize_gold_from_review.py"),
            "--csv", str(drafts / "gold_review_queue_local.csv"), "--outdir", temporary,
        ], check=True, capture_output=True)
        generated = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in Path(temporary).glob("*.json")}
    stored = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in (drafts / "gold").glob("*.json")}
    if not generated or generated != stored:
        raise ValueError("Checked review does not reproduce saved gold; manual decision required")
    plan.gold_provenance = {
        "review_csv": "outputs/gold_draft_legacy/gold_review_queue_local.csv",
        "review_rows": len(read_csv(drafts / "gold_review_queue_local.csv")),
        "gold_files": len(stored),
        "validation": "Reviewed CSV reproduces saved gold and gold.zip; scene person IDs match.",
    }
    with zipfile.ZipFile(drafts / "gold.zip") as archive:
        zipped = {Path(p.filename).name: json.loads(archive.read(p)) for p in archive.infolist() if not p.is_dir() and p.filename.startswith("gold/") and p.filename.endswith(".json")}
        if zipped != stored:
            raise ValueError("gold.zip differs from checked gold; manual decision required")
        for member in archive.infolist():
            if not member.is_dir():
                plan.add(Path("data/review/legacy") / member.filename, archive.read(member))
    for source in sorted((drafts / "gold").glob("*.json")):
        plan.copy(source.relative_to(root), Path("data/gold") / source.name)
    manifest = json.loads((legacy / "runs/oracle_evidence_002/manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        graph = json.loads(checked_path(root, Path("experiments/runs/oracle_evidence_002/scene_graph") / f"{item['id']}.json").read_text(encoding="utf-8"))
        if {p["person_id"] for p in graph["persons"]} != {p["person_id"] for p in stored[f"{item['id']}.json"]["persons"]}:
            raise ValueError(f"Gold and scene person IDs differ: {item['id']}")

    for source in sorted(drafts.glob("*.csv")):
        rows = read_csv(source)
        for row in rows:
            for key in ("image_path", "review_image"):
                if row.get(key):
                    row[key] = str(checked_path(root, mapped_path(row[key])))
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        plan.add(Path("outputs/gold_draft_legacy") / source.name, output.getvalue().encode("utf-8-sig"))
    for source in sorted((drafts / "draft_json").glob("*.json")):
        value = json.loads(source.read_text(encoding="utf-8"))
        for key in ("image_path", "numbered_visualization"):
            if value.get(key):
                value[key] = str(checked_path(root, mapped_path(value[key])))
        plan.add(Path("outputs/gold_draft_legacy/draft_json") / source.name, json_bytes(value))
    return plan


def verify(root: Path):
    resolved_directory.cache_clear()
    report = json.loads((root / REPORT).read_text(encoding="utf-8"))
    if digest(root / INVENTORY) != report["inventory_sha256"]:
        raise ValueError("Migration inventory hash mismatch")
    with (root / INVENTORY).open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    archived_sources = set()
    for index, row in enumerate(records, 1):
        target = checked_path(root, Path(row["destination"]))
        if not target.is_file() or target.stat().st_size != int(row["bytes"]) or digest(target) != row["sha256"]:
            raise ValueError(f"Missing or changed migrated file: {target}")
        if row["source"]:
            source = checked_path(root, Path(row["source"]))
            if source.exists() and digest(source) != row["sha256"]:
                raise ValueError(f"Source changed after migration: {source}")
            if row["source"].startswith("experiments/"):
                archived_sources.add(row["source"])
        if index % 5000 == 0:
            print(f"Verified inventory: {index}/{len(records)} files", flush=True)
    if (root / "experiments").exists():
        current = {p.relative_to(root).as_posix() for p in (root / "experiments").rglob("*") if p.is_file()}
        if current != archived_sources:
            raise ValueError("experiments file inventory changed; do not delete it before reconciling")
    for manifest in ("data/evidence/oracle/manifest.json", "data/evidence/oracle/manifest_with_gold.json", "data/evidence/oracle_001/manifest.json"):
        samples = load_manifest(root / manifest)
        validate_samples(samples, make_conditions("leave_one_out"))
        for sample in samples:
            for key in ("original_image", "numbered_visualization", "scene_graph", "rule_result", "gold"):
                path = getattr(sample, key)
                if path and (not path.is_relative_to(root) or path.is_relative_to(root / "experiments") or path.is_relative_to(root / "input_demo")):
                    raise ValueError(f"Unmigrated working path: {path}")
    print(f"Verified {len(records)} files; {len(archived_sources)} experiments originals archived exactly.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Copy after complete conflict preflight")
    mode.add_argument("--verify", action="store_true", help="Verify saved hashes, including after experiments removal")
    args = parser.parse_args()
    if args.verify:
        verify(ROOT)
        return
    plan = build_plan(ROOT)
    print(f"Plan: {len(plan.entries)} files, {sum(row['bytes'] for row in plan.entries.values()):,} bytes. No source deletion.")
    if not args.apply:
        print("Preview only. Use --apply to copy, then --verify before manual cleanup.")
        return
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["source", "destination", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(plan.entries.values())
    inventory = output.getvalue().encode("utf-8")
    report = {
        "policy": "Copy only; existing different files rejected before writes; originals untouched.",
        "files": len(plan.entries), "bytes": sum(row["bytes"] for row in plan.entries.values()),
        "inventory": INVENTORY.as_posix(), "inventory_sha256": hashlib.sha256(inventory).hexdigest(),
        "gold_provenance": plan.gold_provenance,
        "archive": "outputs/legacy; historical files retain original bytes and paths as provenance.",
    }
    plan.add(INVENTORY, inventory)
    plan.add(REPORT, json_bytes(report))
    plan.apply()
    verify(ROOT)


if __name__ == "__main__":
    main()
