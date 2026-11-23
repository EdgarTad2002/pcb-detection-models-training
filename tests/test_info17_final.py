import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.prepare_info17_final import convert_label, build_original
from tools.controlled_experiments import make_jobs, command_for, read_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_original_class_filter_never_merges_capacitor_jumper():
    rows, polygons = convert_label("1 0.5 0.5 0.1 0.2\n2 0.3 0.4 0.1 0.2\n9 0.5 0.5 0.2 0.3\n")
    assert [r.split()[0] for r in rows] == ["0", "3"]
    assert [float(x) for x in rows[0].split()[1:]] == [0.3, 0.4, 0.1, 0.2]
    assert polygons == []


def test_polygon_becomes_tight_bbox_without_losing_class():
    rows, polygons = convert_label("2 0.1 0.2 0.7 0.2 0.7 0.8 0.1 0.8\n")
    assert polygons == [1]
    assert rows[0].split()[0] == "0"
    assert [float(x) for x in rows[0].split()[1:]] == pytest.approx([0.4, 0.5, 0.6, 0.6])
    with pytest.raises(ValueError):
        convert_label("2 0.5 nan 0.2 0.2")


def test_dataset_preserves_source_and_refuses_overwrite(tmp_path):
    import yaml
    from PIL import Image
    source, out = tmp_path / "raw", tmp_path / "derived"
    source.mkdir()
    names = ["unused"] * 23
    for idx, name in {2: "Capacitor", 4: "Connector", 7: "Electrolytic Capacitor", 9: "IC"}.items():
        names[idx] = name
    (source / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    for i, split in enumerate(("train", "valid", "test")):
        images, labels = source / split / "images", source / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        Image.new("RGB", (640, 640), (i * 40, 60, 100)).save(images / f"board_{split}.jpg")
        (labels / f"board_{split}.txt").write_text("2 0.5 0.5 0.1 0.1\n")
        Image.new("RGB", (640, 640)).save(images / "background.jpg")
        (labels / "background.txt").write_text("17 0.5 0.5 0.1 0.1\n")
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result = build_original(source, out)
    assert result["stats"]["train"]["retained_images"] == 1
    assert len(read_dataset(out / "data.yaml")["splits"]["val"]) == 1
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        build_original(source, out)


def test_eighteen_runs_and_matched_crop_budget():
    config = json.loads((ROOT / "configs/info17_final_experiments.json").read_text())
    jobs = make_jobs(config, ["main", "crop"], counts={"paper": 543, "mixed": 1086})
    assert len(jobs) == 18
    for job in jobs:
        assert job["seed"] in (0, 1, 2)
        args = command_for(config, job, ROOT, Path("/tmp/info17"), Path("data.yaml"), Path("eval.yaml"))
        assert args[args.index("--eval-split") + 1] == "val"
        assert "ensemble_champions" not in " ".join(args)
        assert config["models"][job["model"]]["weights"] in {"yolo26s.pt", "yolo12s.pt", "yolov8s.pt"}
        if job["budget"]:
            assert job["epochs"] == 50
            assert job["budget"]["planned_batches"] == 3400
            assert job["budget"]["relative_difference"] == 0
        else:
            assert job["epochs"] == 100
    assert not config["models"]["yolov12s"].get("squared_dice")
    assert not config["models"]["physot_yolo26s"].get("squared_dice")
    for name in ("yolo26s_agwd_sdice", "physot_yolo26s", "yolo26s_crop_specialist"):
        assert config["models"][name]["agwd_kappa"] == 6.4
        assert config["models"][name]["agwd_tau"] == 144


def test_agwd_dice_excludes_inactive_channels_in_loss_and_gradient():
    import torch
    from agwd_ota_loss import AGWDDetectionLoss
    torch.set_num_threads(2)
    torch.manual_seed(13)
    scores = torch.randn(2, 20, 23, requires_grad=True)
    targets = torch.zeros_like(scores)
    targets[0, 3, 2] = 1
    targets[1, 7, 9] = 0.7
    feats = [torch.zeros(2, 1, 4, 4), torch.zeros(2, 1, 2, 2)]
    settings = SimpleNamespace(smooth_dice=0.01, stage_weights=None)
    actual = AGWDDetectionLoss._compute_multistage_sdice(settings, scores, targets, feats)
    expected = AGWDDetectionLoss._compute_multistage_sdice(settings, scores[..., [2, 4, 7, 9]], targets[..., [2, 4, 7, 9]], feats)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    inactive = [i for i in range(23) if i not in (2, 4, 7, 9)]
    assert torch.count_nonzero(scores.grad[..., inactive]) == 0
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[..., [2, 4, 7, 9]].abs().sum() > 0
    changed = scores.detach().clone()
    changed[..., inactive] = 50
    torch.testing.assert_close(actual.detach(), AGWDDetectionLoss._compute_multistage_sdice(settings, changed, targets, feats))


def test_scale_gate_and_wasserstein_constant_at_half_resolution():
    import torch
    from agwd_ota_loss import compute_scale_gate_psi, compute_nwd_similarity
    gt = torch.tensor([[0., 0., 24., 24.], [0., 0., 80., 80.]])
    pred = gt + 2
    torch.testing.assert_close(compute_scale_gate_psi(gt, 576), compute_scale_gate_psi(gt / 2, 144))
    torch.testing.assert_close(compute_nwd_similarity(pred, gt, constant_c=12.8, eta=0),
                               compute_nwd_similarity(pred / 2, gt / 2, constant_c=6.4, eta=0))
