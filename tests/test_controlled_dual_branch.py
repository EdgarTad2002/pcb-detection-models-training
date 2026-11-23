"""Regression tests for preserving global detections and independent tile/input sizes."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tools.eval_dual_branch_pipeline import run_dual_branch_inference, fuse_capacitor_predictions_wbf


def box(cid, score, coords):
    return SimpleNamespace(cls=torch.tensor([cid]), conf=torch.tensor([score]), xyxy=torch.tensor([coords]))


class Model:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image.shape, kwargs))
        return [SimpleNamespace(boxes=self.boxes)]


class ControlledDualBranchTest(unittest.TestCase):
    def infer(self, global_model, crop_model, max_det=1000, merge_iou=0.4):
        return run_dual_branch_inference(global_model, crop_model, np.zeros((100, 120, 3), dtype=np.uint8),
                                         slice_size=64, overlap=0.2, conf_thresh=0.001, merge_iou=merge_iou,
                                         ios_thresh=0.65, max_det=max_det, fusion_method="nms", device="cpu",
                                         imgsz=1280, prediction_iou=0.5)

    def test_merge_sweep_does_not_suppress_global_macro_predictions(self):
        global_model = Model([box(1, 0.2, [0., 0., 30., 30.]), box(1, 0.3, [1., 1., 31., 31.])])
        crop_model = Model([])
        dets, count = self.infer(global_model, crop_model)
        self.assertEqual(len(dets), 2)
        self.assertGreater(count, 1)
        self.assertEqual(global_model.calls[0][1]["iou"], 0.5)
        self.assertEqual(global_model.calls[0][1]["imgsz"], 1280)
        self.assertTrue(all(c[1]["imgsz"] == 1280 for c in crop_model.calls))
        self.assertTrue(all(c[0][0] <= 64 and c[0][1] <= 64 for c in crop_model.calls))

    def test_crop_macro_predictions_are_discarded_and_global_slots_reserved(self):
        global_model = Model([box(3, 0.1, [1., 1., 10., 10.])])
        crop_model = Model([box(0, 0.9, [20., 20., 25., 25.]), box(1, 0.99, [1., 1., 10., 10.])])
        dets, _ = self.infer(global_model, crop_model, max_det=2)
        self.assertEqual(len(dets), 2)
        self.assertEqual(sorted(d[0] for d in dets), [0, 3])

    def test_wbf_does_not_silently_fall_back_to_nms(self):
        with patch.dict("sys.modules", {"ensemble_boxes": None}):
            with self.assertRaisesRegex(RuntimeError, "Refusing to report NMS"):
                fuse_capacitor_predictions_wbf(torch.tensor([[0., 0., 5., 5.], [1., 1., 6., 6.]]),
                                               torch.tensor([0.9, 0.8]), 100, 100)


if __name__ == "__main__":
    unittest.main()
