"""Numerical and input-integrity checks for database group means."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from general_sers_analysis.dataset_visualization import load_group_means, visualize_dataset


class GroupMeanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with (self.root / "content.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Data_id", "task", "modality", "n_group", "n_spectra"])
            writer.writerow(["TEST", "Classification", "Raman", 2, 3])
        self.group = self.root / "data" / "TEST" / "Class_1_0_1"
        self.group.mkdir(parents=True)
        other = self.group.parent / "Class_0_1_2"
        other.mkdir()
        self.write(self.group / "one.csv", [[400, 1], [401, 3], [402, 5]])
        self.write(self.group / "two.csv", [[402, 9], [401, 7], [400, 3]])
        self.write(other / "one.csv", [[400, 10], [401, 20], [402, 30]])

    def write(self, path, values):
        np.savetxt(path, values, delimiter=",", header="RamanShift,intensity", comments="")

    def test_exact_means_group_identity_and_single_repeat(self):
        _, means = load_group_means(self.root, "TEST")
        by_name = {group.group_id: group for group in means}
        repeat = by_name[self.group.name]
        np.testing.assert_allclose(repeat.intensity, [2, 5, 7])
        self.assertEqual(repeat.n_spectra, 2)
        np.testing.assert_allclose(by_name["Class_0_1_2"].intensity, [10, 20, 30])

    def test_axis_mismatch_rejected_without_interpolation(self):
        self.write(self.group / "two.csv", [[400, 3], [401.5, 7], [402, 9]])
        with self.assertRaisesRegex(ValueError, "RamanShift mismatch"):
            load_group_means(self.root, "TEST")

    def test_selection_unknown_id_and_invalid_values(self):
        _, means = load_group_means(self.root, "TEST", groups=[self.group.name])
        self.assertEqual(len(means), 1)
        for data_id in ("UNKNOWN", "../TEST"):
            with self.assertRaises(ValueError):
                load_group_means(self.root, data_id)
        with self.assertRaisesRegex(ValueError, "Unknown group"):
            load_group_means(self.root, "TEST", groups=["missing"])
        self.write(self.group / "two.csv", [[400, 3], [401, np.nan], [402, 9]])
        with self.assertRaisesRegex(ValueError, "Invalid spectrum"):
            load_group_means(self.root, "TEST")

    def test_headless_figure_is_written(self):
        output, means = visualize_dataset(self.root, "TEST", self.root / "figures")
        self.assertEqual(len(means), 2)
        self.assertTrue(output.is_file())
        self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
