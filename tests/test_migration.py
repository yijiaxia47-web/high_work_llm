"""Checks for migration data-loss prevention; no real data or API required."""
from pathlib import Path
import tempfile
import unittest

from scripts.migrate_experiments import Plan, checked_path, mapped_path, rebase_manifest


class MigrationTests(unittest.TestCase):
    def test_conflict_preflight_leaves_all_files_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "existing").write_bytes(b"keep this")
            plan = Plan(root)
            plan.add(Path("new"), b"new content")
            with self.assertRaises(FileExistsError):
                plan.add(Path("existing"), b"different content")
            self.assertFalse((root / "new").exists())
            self.assertEqual((root / "existing").read_bytes(), b"keep this")

    def test_copy_is_exact_idempotent_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = bytes(range(256))
            (root / "source").write_bytes(payload)
            for _ in range(2):
                plan = Plan(root)
                plan.copy(Path("source"), Path("nested/copy"))
                plan.apply()
            self.assertEqual((root / "source").read_bytes(), payload)
            self.assertEqual((root / "nested/copy").read_bytes(), payload)

    def test_destination_changed_after_preflight_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = Plan(root)
            plan.add(Path("target"), b"planned")
            (root / "target").write_bytes(b"another writer")
            with self.assertRaises(FileExistsError):
                plan.apply()
            self.assertEqual((root / "target").read_bytes(), b"another writer")

    def test_rebase_handles_foreign_roots_and_retains_sample_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            old = "Z:/previous/project/experiments/runs/oracle_evidence_002/scene_graph/a.json"
            self.assertEqual(mapped_path(old), Path("data/evidence/oracle/scene_graph/a.json"))
            value = [{"id": "a", "scene_graph": old, "annotation": "keep"}]
            result = rebase_manifest(value, Path("data/evidence/oracle/manifest.json"), root)
            self.assertEqual(result, [{"id": "a", "scene_graph": "scene_graph/a.json", "annotation": "keep"}])
            with self.assertRaises(ValueError):
                mapped_path("Z:/unknown/a.json")
            with self.assertRaises(ValueError):
                checked_path(root, Path("../outside"))
            with self.assertRaises(ValueError):
                checked_path(root, Path(".."))

    def test_symlink_cannot_redirect_copy_outside_project(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            destination = Path(outside) / "untouched"
            destination.write_bytes(b"keep")
            try:
                (root / "link").symlink_to(destination)
            except OSError:
                self.skipTest("Creating symlinks is unavailable on this host")
            with self.assertRaises(ValueError):
                checked_path(root, Path("link"))
            self.assertEqual(destination.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
