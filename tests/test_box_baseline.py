"""Geometry and matching checks for the heuristic box baseline."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
from box_baseline import corners, evaluate, iou3d, make_boxes


class BoxBaselineTests(unittest.TestCase):
    def test_iou_and_corners(self):
        box = [0, 0, 0, 2, 2, 2, 0]
        self.assertAlmostEqual(iou3d(box, box), 1)
        self.assertAlmostEqual(iou3d(box, [1, 0, 0, 2, 2, 2, 0]), 1 / 3)
        self.assertEqual(iou3d(box, [3, 0, 0, 2, 2, 2, 0]), 0)
        np.testing.assert_allclose(corners(box).min(0), [-1, -1, -1])
        np.testing.assert_allclose(corners(box).max(0), [1, 1, 1])

    def test_matching_is_one_to_one_and_class_aware(self):
        box = [0, 0, 0, 2, 2, 2, 0]
        references = [dict(id=0, label="chair", box=box)]
        predicted = [dict(id=0, label="table", box=box, score=.99),
                     dict(id=1, label="chair", box=box, score=.9),
                     dict(id=2, label="chair", box=box, score=.8)]
        result = evaluate(predicted, references, .5)
        self.assertEqual((result["true_positive"], result["false_positive"], result["false_negative"]), (1, 2, 0))
        self.assertEqual(result["matches"][0]["prediction_id"], 1)
        self.assertEqual(evaluate([], references, .5)["false_negative"], 1)

    def test_clustering_separates_instances_and_excludes_wall(self):
        grid = np.stack(np.meshgrid(*[np.arange(4) * .03] * 3), -1).reshape(-1, 3)
        coord = np.concatenate([grid, grid + [1, 0, 0], grid + [2, 0, 0]])
        labels = np.repeat([2, 2, 0], len(grid))
        confidence = np.ones(len(coord))
        args = SimpleNamespace(eps=.051, min_samples=3, min_points=20, min_confidence=.45)
        names = ["wall", "floor", "cabinet"] + [str(i) for i in range(17)]
        boxes, assignment = make_boxes(coord, labels, confidence, names, args)
        self.assertEqual(len(boxes), 2)
        self.assertTrue(np.all(assignment[-len(grid):] == -1))
        for box in boxes:
            points = coord[assignment == box["id"]]
            np.testing.assert_allclose(points.min(0), box["min_xyz"])
            np.testing.assert_allclose(points.max(0), box["max_xyz"])


if __name__ == "__main__":
    unittest.main()
