"""Independent dataset metadata and class-mismatch safety checks."""
from pathlib import Path
import tempfile
import unittest

import yaml

from main.evidence import CLASS_NAMES
from scripts.migrate_harness_dataset import dataset_yaml, inspect_dataset
from scripts.prepare_evidence_from_yolo_labels import validate_dataset_classes


class HarnessDatasetTests(unittest.TestCase):
    def make_dataset(self, root):
        (root / "labels").mkdir()
        (root / "labels/labels.txt").write_text("aqd_cdgf\naqd_gfsy\naqd_wpd\naqd_wzqpd", encoding="utf-8")
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir()
            (root / "images" / split / "a.jpg").write_bytes(b"fixture")
            (root / "labels" / split / "a.txt").write_text("3 0.5 0.5 0.2 0.3\n", encoding="utf-8")

    def test_own_classes_and_val_split_remain_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_dataset(root)
            metadata = inspect_dataset(root)
            config = yaml.safe_load(dataset_yaml(metadata["names"]))
            self.assertEqual(config["names"], ["aqd_cdgf", "aqd_gfsy", "aqd_wpd", "aqd_wzqpd"])
            self.assertEqual(config["val"], "images/val")
            self.assertEqual(config["nc"], 4)
            self.assertEqual(metadata["split_pairs"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(metadata["objects_by_class_id"]["3"], 3)

    def test_missing_label_and_invalid_class_stop_migration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_dataset(root)
            label = root / "labels/train/a.txt"
            label.write_text("4 0.5 0.5 0.2 0.3", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid YOLO"):
                inspect_dataset(root)
            label.unlink()
            with self.assertRaisesRegex(ValueError, "not paired"):
                inspect_dataset(root)

    def test_six_class_pipeline_rejects_four_classes_and_swapped_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for names in (["aqd_cdgf", "aqd_gfsy", "aqd_wpd", "aqd_wzqpd"], list(reversed(CLASS_NAMES))):
                (root / "data.yaml").write_bytes(dataset_yaml(names))
                with self.assertRaisesRegex(ValueError, "do not match"):
                    validate_dataset_classes(root)
            (root / "data.yaml").write_bytes(dataset_yaml(CLASS_NAMES))
            validate_dataset_classes(root)


if __name__ == "__main__":
    unittest.main()
