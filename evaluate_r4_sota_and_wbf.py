#!/usr/bin/env python3
"""
Evaluates all R4-Final models on the held-out Test Split (44 images, 4,909 targets)
with TRUE per-class AP@50 (metrics.box.ap50) AND runs Multi-Model Weighted Boxes Fusion (WBF).
"""
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

for mod in ["agwd_ota_loss", "squared_dice_loss", "retinex_stem", "weber_stem", "train_sota_r4"]:
    try:
        __import__(mod)
    except Exception:
        pass

from train_sota_r4 import install_retinex_sargis_hook

DATA_YAML = "datasets/pcb-native-reviewed-r4-final/data.yaml"
OUT_JSON = Path("results/r4_final_sota_research_summary.json")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    ("SOTA_YOLO11m_RetinexEdge_AGWD_1536", "runs/r4_sota_yolo11m_retinex_agwd_1536/pcb-filtered/weights/best.pt", 1536, True, 1.40),
    ("SOTA_YOLO26x_RetinexEdge_AGWD_1280", "runs/r4_sota_yolo26x_retinex_agwd_1280/pcb-filtered/weights/best.pt", 1280, True, 1.30),
    ("SOTA_YOLO26s_RetinexEdge_AGWD_1536", "runs/r4_sota_yolo26s_retinex_agwd_1536/pcb-filtered/weights/best.pt", 1536, True, 1.25),
    ("Baseline_YOLO11m_200ep_1280", "runs/detect/runs/r4_baseline_yolo11m_multiclass_200ep/weights/best.pt", 1280, False, 1.10),
    ("Prev_YOLO26s_AGWD_SDice_1280", "runs/r4_final_yolo26s_agwd_sdice_1280/pcb-filtered/weights/best.pt", 1280, False, 1.00),
]


def wbf_single_class(boxes_list, scores_list, weights, iou_thr=0.55):
    """Pure PyTorch/NumPy Weighted Boxes Fusion for a single class."""
    all_b, all_s, all_w = [], [], []
    total_w = float(sum(weights))
    for b, s, w in zip(boxes_list, scores_list, weights):
        if len(b) == 0:
            continue
        all_b.append(b)
        all_s.append(s * float(w))
        all_w.append(np.full(len(s), float(w), dtype=np.float32))
    if not all_b:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    all_b = np.concatenate(all_b, axis=0)
    all_s = np.concatenate(all_s, axis=0)
    all_w = np.concatenate(all_w, axis=0)
    order = np.argsort(-all_s)
    all_b, all_s, all_w = all_b[order], all_s[order], all_w[order]

    clusters_b, clusters_s, clusters_w, fused_b = [], [], [], []

    for b, s, w in zip(all_b, all_s, all_w):
        if not fused_b:
            clusters_b.append([b])
            clusters_s.append([s])
            clusters_w.append([w])
            fused_b.append(b.copy())
            continue
        fb = np.stack(fused_b, axis=0)
        ix1 = np.maximum(fb[:, 0], b[0])
        iy1 = np.maximum(fb[:, 1], b[1])
        ix2 = np.minimum(fb[:, 2], b[2])
        iy2 = np.minimum(fb[:, 3], b[3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_a = (fb[:, 2] - fb[:, 0]) * (fb[:, 3] - fb[:, 1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        ious = inter / np.maximum(area_a + area_b - inter, 1e-7)
        best_idx = int(np.argmax(ious))
        if ious[best_idx] >= iou_thr:
            clusters_b[best_idx].append(b)
            clusters_s[best_idx].append(s)
            clusters_w[best_idx].append(w)
            cs = np.array(clusters_s[best_idx], dtype=np.float32)
            cb = np.stack(clusters_b[best_idx], axis=0)
            fused_b[best_idx] = np.sum(cb * cs[:, None], axis=0) / np.maximum(np.sum(cs), 1e-7)
        else:
            clusters_b.append([b])
            clusters_s.append([s])
            clusters_w.append([w])
            fused_b.append(b.copy())

    out_b = np.stack(fused_b, axis=0)
    out_s = np.array(
        [np.sum(cs) / total_w for cs in clusters_s],
        dtype=np.float32,
    )
    order = np.argsort(-out_s)
    return out_b[order], out_s[order]


def main():
    install_retinex_sargis_hook(ret_alpha=0.10, edge_alpha=0.08)
    summary = {}
    available_models = []

    print("\n================ SINGLE-MODEL EVALUATION ON R4-FINAL TEST ================")
    for name, w_path, imgsz, use_ret, w_ens in CANDIDATES:
        p = Path(w_path)
        if not p.exists():
            print(f"[SKIP] {name}: {w_path} not found")
            continue
        model = YOLO(str(p))
        metrics = model.val(
            data=DATA_YAML,
            split="test",
            imgsz=imgsz,
            conf=0.001,
            iou=0.50,
            max_det=1000,
            device=0,
            plots=False,
            verbose=False,
        )
        per_class_ap50 = {}
        per_class_ap50_95 = {}
        for idx, cls_idx in enumerate(metrics.box.ap_class_index):
            cname = metrics.names[int(cls_idx)]
            per_class_ap50[cname] = float(metrics.box.ap50[idx])
            per_class_ap50_95[cname] = float(metrics.box.maps[idx])

        res = {
            "weights": str(p),
            "imgsz": imgsz,
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "per_class_ap50": per_class_ap50,
            "per_class_ap50_95": per_class_ap50_95,
        }
        summary[name] = res
        available_models.append((name, model, imgsz, w_ens))
        print(
            f"{name:38s} | mAP50={res['mAP50']:.4f} | mAP50-95={res['mAP50_95']:.4f} | "
            f"P={res['precision']:.4f} | R={res['recall']:.4f} | AP50={per_class_ap50}"
        )

    # Multi-Model WBF Ensemble across available models
    if len(available_models) >= 2:
        print("\n================ MULTI-MODEL WBF ENSEMBLE ON R4-FINAL TEST ================")
        cfg = yaml.safe_load(open(DATA_YAML))
        root = Path(cfg["path"])
        test_imgs = sorted((root / cfg["test"]).glob("*"))
        names = cfg["names"]
        nc = len(names)
        iouv = torch.linspace(0.5, 0.95, 10)
        stats = []

        weights = [m[3] for m in available_models]
        for im_path in test_imgs:
            W, H = Image.open(im_path).size
            lb_path = root / "test" / "labels" / (im_path.stem + ".txt")
            gt_cls, gt_boxes = [], []
            if lb_path.exists():
                for line in lb_path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    c = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = (cx - bw / 2.0) * W
                    y1 = (cy - bh / 2.0) * H
                    x2 = (cx + bw / 2.0) * W
                    y2 = (cy + bh / 2.0) * H
                    gt_cls.append(c)
                    gt_boxes.append([x1, y1, x2, y2])
            gt_cls_t = torch.tensor(gt_cls, dtype=torch.int64)
            gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4))

            per_model_preds = []
            for _, model, imgsz, _ in available_models:
                r = model.predict(str(im_path), imgsz=imgsz, conf=0.001, iou=0.55, max_det=1000, verbose=False)[0]
                per_model_preds.append((
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy().astype(int),
                ))

            fused_xyxy, fused_conf, fused_cls = [], [], []
            for c in range(nc):
                b_list = [p[0][p[2] == c] for p in per_model_preds]
                s_list = [p[1][p[2] == c] for p in per_model_preds]
                fb, fs = wbf_single_class(b_list, s_list, weights, iou_thr=0.55)
                if len(fb) > 0:
                    fused_xyxy.append(fb)
                    fused_conf.append(fs)
                    fused_cls.append(np.full(len(fs), c, dtype=np.int64))

            if fused_xyxy:
                pred_b = torch.from_numpy(np.concatenate(fused_xyxy, axis=0)).float()
                pred_s = torch.from_numpy(np.concatenate(fused_conf, axis=0)).float()
                pred_c = torch.from_numpy(np.concatenate(fused_cls, axis=0)).long()
                order = torch.argsort(pred_s, descending=True)[:1000]
                pred_b, pred_s, pred_c = pred_b[order], pred_s[order], pred_c[order]
            else:
                pred_b = torch.zeros((0, 4))
                pred_s = torch.zeros((0,))
                pred_c = torch.zeros((0,), dtype=torch.int64)

            if len(pred_b) == 0:
                if len(gt_cls_t) > 0:
                    stats.append((torch.zeros((0, 10), dtype=torch.bool), torch.zeros((0,)), torch.zeros((0,)), gt_cls_t))
                continue

            if len(gt_cls_t) > 0:
                ious = box_iou(gt_boxes_t, pred_b)
                correct = np.zeros((len(pred_b), len(iouv)), dtype=bool)
                correct_class = gt_cls_t[:, None] == pred_c
                for i, thr in enumerate(iouv):
                    x = torch.where((ious >= thr) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), ious[x[0], x[1]][:, None]), 1).cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), i] = True
                tp = torch.tensor(correct, dtype=torch.bool)
            else:
                tp = torch.zeros((len(pred_b), 10), dtype=torch.bool)

            stats.append((tp, pred_s, pred_c, gt_cls_t))

        tp, conf, pred_cls, target_cls = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
        tp_r, fp_r, p, r, f1, ap, ap_class = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
        ap50, ap50_95 = ap[:, 0], ap.mean(1)
        ens_res = {
            "models_fused": [m[0] for m in available_models],
            "mAP50": float(ap50.mean()),
            "mAP50_95": float(ap50_95.mean()),
            "precision": float(p.mean()),
            "recall": float(r.mean()),
            "per_class_ap50": {names[int(c)]: float(ap50[i]) for i, c in enumerate(ap_class)},
            "per_class_ap50_95": {names[int(c)]: float(ap50_95[i]) for i, c in enumerate(ap_class)},
        }
        summary["WBF_Super_Ensemble_R4_Test"] = ens_res
        print(
            f"WBF_Super_Ensemble_R4_Test             | mAP50={ens_res['mAP50']:.4f} | "
            f"mAP50-95={ens_res['mAP50_95']:.4f} | P={ens_res['precision']:.4f} | "
            f"R={ens_res['recall']:.4f} | AP50={ens_res['per_class_ap50']}"
        )

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved full benchmark summary to {OUT_JSON}")


if __name__ == "__main__":
    main()
