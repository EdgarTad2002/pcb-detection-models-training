import json

import pytest

from tools.evaluate_info17_protocol import (
    AUTHOR_SOURCE, TO_AUTHOR, aggregate, author_inputs, checked_author_source,
    checkpoint_entries, load_metric_functions,
)


def test_class_mapping_preserves_semantics_and_boxes():
    records = [{"name": "a.jpg", "boxes_cxywh": [[3, .5, .5, .2, .4]]}]
    p, g, pn, gn = author_inputs([[[256, 192, 384, 448, .9, 3]]], records)
    assert TO_AUTHOR == {0: 1, 1: 2, 2: 3, 3: 0}
    assert p[0]["class_id"] == g[0]["class_id"] == 0
    assert pn["a.jpg"][0]["box_norm"] == pytest.approx([.4, .3, .6, .7])
    assert gn["a.jpg"][0]["box_norm"] == pytest.approx([.4, .3, .6, .7])


def test_public_evaluators_never_match_detections_across_images():
    functions = load_metric_functions()
    predictions = [{"image_id": "a", "class_id": 0, "box_xyxy": [0., 0., 10., 10.], "score": .9}]
    ground_truth = [{"image_id": "b", "class_id": 0, "box_xyxy": [0., 0., 10., 10.]}]
    assert functions["evaluate_predictions"](predictions, ground_truth)["mAP50"] == 0
    # Upstream 11-point evaluator assigns 1/11 AP via its leading sentinel,
    # even with no matching detection. Preserve and disclose this behavior.
    value = functions["compute_map"](
        {"a": [{"class": 0, "confidence": .9, "box_norm": [0, 0, 1, 1]}]},
        {"b": [{"class": 0, "box_norm": [0, 0, 1, 1]}]}, ["IC"], [0.5])
    assert value["mAP50"] == pytest.approx(1 / 11)


def test_public_nms_iou_tie_is_not_confidence_prioritized():
    m = load_metric_functions()
    p = [{"image_id": "a", "class_id": 0, "box_xyxy": [0., 0., 10., 10.], "score": .9},
         {"image_id": "a", "class_id": 0, "box_xyxy": [0., 0., 10., 10.], "score": .1}]
    g = [{"image_id": "a", "class_id": 0, "box_xyxy": [0., 0., 10., 10.]}]
    # Upstream matching sorts IoU ties, not detection confidence; for this
    # exact tie the low-confidence duplicate is assigned the GT. Preserve it.
    assert m["evaluate_predictions"](p, g)["mAP50"] == pytest.approx(.4975)
    assert m["evaluate_predictions"](p[:1], g)["mAP50"] == pytest.approx(.995)


def test_aggregate_requires_all_seeds_and_uses_sample_std():
    rows = [dict(view="v", split="test", model="m", seed=i, mAP50=v)
            for i, v in enumerate([.4, .5, .6])]
    summary = aggregate(rows)[0]
    assert summary["mAP50_mean"] == pytest.approx(.5)
    assert summary["mAP50_std"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="Incomplete seed"):
        aggregate(rows[:2])


def test_reject_changed_checkpoint_before_inference(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"modified")
    frozen = {"checkpoints": [{"job": "j", "weights": str(weights), "sha256": "0" * 64}]}
    plan = {"jobs": [{"key": "j", "model": "yolov8s", "seed": 0}], "config": {"models": ["yolov8s"]}}
    with pytest.raises(ValueError, match="Checkpoint changed"):
        checkpoint_entries(plan, frozen)


def test_all_pinned_author_files_verified():
    source = checked_author_source()
    assert source["commit"] == "21044102c2d73e516d45e17a7fc06e894e841104"
    assert len(source["files"]) == 11
