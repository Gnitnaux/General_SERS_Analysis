import tempfile
from pathlib import Path
import unittest
import numpy as np
from general_sers_analysis import Node, Spectra
from general_sers_analysis.tools.visualization import GroupSpectrumPlot, PredictionPlot, ConfusionMatrixPlot

class VisualizationNodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "figure.png"

    def check_png(self):
        self.assertEqual(self.path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_group_plot_preserves_input(self):
        x = Spectra([[1,2,3],[3,4,5],[0,1,0]], [400,401,402])
        node = GroupSpectrumPlot(self.path)
        self.assertIsInstance(node, Node)
        self.assertIs(node.transform(x, groups=["a","a","b"]), x)
        self.assertEqual(node.group_ids_, ("a", "b"))
        self.check_png()

    def test_prediction_group_metrics_and_multiple_targets(self):
        pred = np.array([[1,4],[3,6],[5,10]])
        truth = np.array([[2,5],[2,5],[5,10]])
        node = PredictionPlot(self.path, molecule_names=["A","B"], unit="ppm")
        self.assertIs(node.transform(pred, y_true=truth, groups=["a","a","b"]), pred)
        self.assertEqual(len(node.metrics_),2)
        self.assertEqual(node.metrics_[0]["RMSE"],0)
        self.check_png()

    def test_constant_single_target_has_no_fake_r2(self):
        node = PredictionPlot(self.path)
        node.transform([0,1], y_true=[0,0])
        self.assertIsNone(node.metrics_[0]["R2"])

    def test_confusion_class_order_and_orientation(self):
        node = ConfusionMatrixPlot(self.path, labels=["B","A"])
        pred = ["A","B","B"]
        self.assertIs(node.transform(pred, y_true=["A","A","B"]), pred)
        np.testing.assert_array_equal(node.matrices_[0], [[1,0],[1,1]])
        self.check_png()

    def test_multilabel_normalization_handles_empty_rows(self):
        node = ConfusionMatrixPlot(self.path, multilabel=True, class_names=["A","B"],normalize="true")
        node.transform([[1,0],[0,0]], y_true=[[1,0],[1,0]])
        np.testing.assert_allclose(node.matrices_[0], [[0,0],[.5,.5]])
        self.assertTrue(np.isfinite(node.matrices_).all())
        self.check_png()

    def test_invalid_shapes_and_missing_classes(self):
        with self.assertRaises(ValueError):
            PredictionPlot(self.path).transform([[1,2]], y_true=[[1]])
        with self.assertRaises(ValueError):
            ConfusionMatrixPlot(self.path,labels=["A"]).transform(["B"],y_true=["A"])
        with self.assertRaises(ValueError):
            ConfusionMatrixPlot(self.path).transform([[.2,.8]],y_true=[[0,1]])
