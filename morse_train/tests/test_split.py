from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import stratified_group_by_file_split


class TestSplit(unittest.TestCase):
    def test_stratified_group_split_keeps_label_coverage(self):
        # 3 labels, 3 files each, 10 segments per file
        y = []
        groups = []
        for lab in [0, 1, 2]:
            for fi in range(3):
                g = f"L{lab}_F{fi}"
                for _ in range(10):
                    y.append(lab)
                    groups.append(g)
        y = np.asarray(y, dtype=np.int64)
        groups = np.asarray(groups)

        train_idx, val_idx, test_idx = stratified_group_by_file_split(
            y,
            groups,
            seed=42,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        self.assertGreater(len(train_idx), 0)
        self.assertGreater(len(val_idx), 0)
        self.assertGreater(len(test_idx), 0)

        train_labels = set(np.unique(y[train_idx]).tolist())
        test_labels = set(np.unique(y[test_idx]).tolist())
        self.assertEqual(train_labels, {0, 1, 2})
        self.assertEqual(test_labels, {0, 1, 2})

        # Group disjointness
        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(val_groups.isdisjoint(test_groups))


if __name__ == "__main__":
    unittest.main()

