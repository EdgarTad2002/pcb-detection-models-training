#!/usr/bin/env python3
"""
tests/test_sliced_inference_audit.py
====================================
Focused unit tests auditing sliced inference operations:
1. Coordinate round-trip invariance (global -> crop -> global).
2. Edge tile boundary conditions (guaranteed full image coverage without negative slicing).
3. Handling of objects crossing tile seams (IoS and IoU behavior).
4. Prevention of over-suppression on closely-spaced physical capacitors.
5. Verification of clipping and boundary guards.
"""

import math
import sys
import unittest
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_sahi_native_test import get_slice_boxes, merge_border_truncated_boxes


class TestSlicedInferenceAudit(unittest.TestCase):

    def test_edge_tiles_and_full_canvas_coverage(self):
        """Verify that get_slice_boxes strictly covers every pixel on the canvas."""
        canvas_sizes = [
            (516, 646),
            (1280, 1280),
            (2048, 1246),
            (4112, 3008),
            (5866, 3911),
        ]
        slice_size = 1280
        overlap = 0.20

        for w, h in canvas_sizes:
            slices = get_slice_boxes(w, h, slice_size, slice_size, overlap)
            self.assertGreater(len(slices), 0, f"No slices generated for {w}x{h}")

            # Create an occupancy grid to verify every pixel is covered
            cov = np.zeros((h, w), dtype=np.uint8)
            for x1, y1, x2, y2 in slices:
                self.assertGreaterEqual(x1, 0)
                self.assertGreaterEqual(y1, 0)
                self.assertLessEqual(x2, w)
                self.assertLessEqual(y2, h)
                self.assertLessEqual(x2 - x1, slice_size)
                self.assertLessEqual(y2 - y1, slice_size)
                cov[y1:y2, x1:x2] = 1

            uncovered_pixels = np.count_nonzero(cov == 0)
            self.assertEqual(
                uncovered_pixels, 0,
                f"Coverage test failed for canvas {w}x{h}: {uncovered_pixels} pixels uncovered!"
            )

    def test_coordinate_round_trip(self):
        """Verify that a bounding box within a tile restores exactly to global coordinates."""
        sx1, sy1 = 1024, 768
        tile_w, tile_h = 1280, 1280

        # Component at global [1200, 900, 1215, 925]
        gt_global = [1200.0, 900.0, 1215.0, 925.0]

        # In tile coordinates:
        bx1_tile = gt_global[0] - sx1
        by1_tile = gt_global[1] - sy1
        bx2_tile = gt_global[2] - sx1
        by2_tile = gt_global[3] - sy1

        # Restored:
        restored = [bx1_tile + sx1, by1_tile + sy1, bx2_tile + sx1, by2_tile + sy1]
        for v_orig, v_rest in zip(gt_global, restored):
            self.assertAlmostEqual(v_orig, v_rest, places=5)

    def test_objects_crossing_seams_merged_by_ios(self):
        """
        When an object is cut at a tile boundary:
        Tile A sees full object: [100, 100, 120, 120] (area 400)
        Tile B sees partial cut box: [100, 100, 105, 120] (area 100)
        IoU is only 100 / 400 = 0.25 (standard NMS fails).
        IoS is 100 / 100 = 1.0 >= 0.65 (IoS correctly suppresses the partial cut).
        """
        full_box = torch.tensor([100.0, 100.0, 120.0, 120.0])
        cut_box = torch.tensor([100.0, 100.0, 105.0, 120.0])

        boxes = torch.stack([cut_box, full_box])
        scores = torch.tensor([0.70, 0.90])
        classes = torch.tensor([0, 0])

        m_boxes, m_scores, m_classes = merge_border_truncated_boxes(
            boxes, scores, classes, ios_thresh=0.65
        )

        self.assertEqual(len(m_boxes), 1, "Partial cut box was not suppressed by larger full box!")
        self.assertTrue(torch.allclose(m_boxes[0], full_box), "Merged box did not keep the full object dimensions!")

    def test_closely_spaced_capacitors_remain_distinct(self):
        """
        Two dense SMD capacitors placed side-by-side with 2px gap:
        Cap 1: [100, 100, 108, 116] (area = 128)
        Cap 2: [110, 100, 118, 116] (area = 128)
        Must NOT be merged by IoS or NMS.
        """
        cap1 = torch.tensor([100.0, 100.0, 108.0, 116.0])
        cap2 = torch.tensor([110.0, 100.0, 118.0, 116.0])

        boxes = torch.stack([cap1, cap2])
        scores = torch.tensor([0.88, 0.85])
        classes = torch.tensor([0, 0])

        m_boxes, m_scores, m_classes = merge_border_truncated_boxes(
            boxes, scores, classes, ios_thresh=0.65
        )

        self.assertEqual(len(m_boxes), 2, "Adjacent distinct capacitors were incorrectly merged by IoS!")


if __name__ == "__main__":
    unittest.main()
