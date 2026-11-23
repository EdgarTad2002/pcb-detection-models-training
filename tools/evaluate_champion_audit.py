#!/usr/bin/env python3
"""Freeze and re-evaluate the exact four champion checkpoints and two WBFs.

No training, threshold search, substituted weights, or fallback fusion algorithm.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.controlled_experiments import NAMES, digest, file_hash, read_dataset, split_signature


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def versions():
    return {p: importlib.metadata.version(p) for p in
            ('torch', 'torchvision', 'ultralytics', 'numpy', 'ensemble-boxes')}


def prepare(config, output):
    if output.exists():
        raise FileExistsError(f'Already prepared: {output}. Use run to resume.')
    cfg = json.loads(config.read_text())
    records = read_dataset(ROOT / cfg['data'])['splits']['test']
    if len(records) != cfg['expected_test_images']:
        raise ValueError('Unexpected test membership; investigate before evaluation')
    models = cfg['models']
    for m in models:
        m['path'] = str((ROOT / m['path']).resolve())
        if file_hash(m['path']) != m['sha256']:
            raise ValueError(f"Checkpoint differs from audited file: {m['id']}")
    paths = [Path(__file__), ROOT / 'tools/benchmark_speed.py', ROOT / 'tools/controlled_experiments.py',
             *ROOT.glob('*.py')]
    manifest = dict(config=cfg, records=records, test_fingerprint=split_signature(records),
                    names=NAMES, versions=versions(), created_utc=datetime.now(timezone.utc).isoformat(),
                    source_hashes={str(p.resolve()): file_hash(str(p.resolve())) for p in paths},
                    disclosure='Historical test results informed development. These are selected checkpoints, not a multi-seed controlled comparison or untouched holdout.',
                    metric='Ultralytics DetectionValidator matching and DetMetrics 101-point interpolated AP at IoU 0.50:0.95',
                    speed_scope='FP32 batch=1, cached decoded test boards, square letterbox, preprocessing + detector + postprocessing + WBF; excludes decoding and model loading')
    manifest['id'] = digest(manifest)
    write(output / 'manifest.json', manifest)
    print(f"Prepared {len(models)} exact checkpoints, 2 ensembles, {len(records)} test boards: {output}", flush=True)


def verify(manifest):
    file_hash.cache_clear()
    if versions() != manifest['versions']:
        raise ValueError('Dependency versions changed after freezing')
    for p, expected in manifest['source_hashes'].items():
        if file_hash(p) != expected:
            raise ValueError(f'Frozen source changed: {p}')
    for m in manifest['config']['models']:
        if file_hash(m['path']) != m['sha256']:
            raise ValueError(f"Frozen checkpoint changed: {m['id']}")
    current = read_dataset(ROOT / manifest['config']['data'])['splits']['test']
    if current != manifest['records']:
        raise ValueError('Test images, labels, or membership changed')


def fuse(predictions, shape, weights, settings):
    import numpy as np
    from ensemble_boxes import weighted_boxes_fusion
    height, width = shape
    scale = np.array([width, height, width, height])
    arrays = [np.asarray(p, dtype=float).reshape(-1, 6) for p in predictions]
    b, s, c = weighted_boxes_fusion(
        [(a[:, :4] / scale).clip(0, 1).tolist() for a in arrays],
        [a[:, 4].tolist() for a in arrays], [a[:, 5].astype(int).tolist() for a in arrays],
        weights=weights, iou_thr=settings['merge_iou'], skip_box_thr=settings['conf'])
    # Preserve WBF's rescaled scores, including values below the member-input floor.
    result = np.column_stack((np.asarray(b).reshape(-1, 4) * scale, s, c))
    return result[np.argsort(-result[:, 4], kind='stable')][:settings['max_det']].tolist()


def score(predictions, records, shapes, settings):
    import numpy as np
    import torch
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils.metrics import DetMetrics
    validator = DetectionValidator(save_dir=Path(records[0]['label']).parent, args={'plots': False})
    validator.iouv = torch.linspace(.5, .95, 10)
    validator.niou = 10
    metrics = DetMetrics(names=dict(enumerate(NAMES)))
    totals = np.zeros(4, dtype=int)
    fixed_tp = np.zeros(4, dtype=int)
    fixed_pred = np.zeros(4, dtype=int)
    for raw, record, shape in zip(predictions, records, shapes, strict=True):
        height, width = shape
        labels = [list(map(float, line.split())) for line in Path(record['label']).read_text().splitlines() if line.strip()]
        labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 5)
        xywh = labels[:, 1:] * torch.tensor([width, height, width, height])
        gt = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)
        pred = torch.tensor(raw, dtype=torch.float32).reshape(-1, 6)
        batch = {'bboxes': gt, 'cls': labels[:, 0]}
        def matched(p):
            return validator._process_batch({'bboxes': p[:, :4], 'cls': p[:, 5]}, batch)['tp']
        metrics.update_stats(dict(tp=matched(pred), conf=pred[:, 4].numpy(), pred_cls=pred[:, 5].numpy(),
                                  target_cls=labels[:, 0].numpy(), target_img=labels[:, 0].unique().numpy(),
                                  im_name=Path(record['label']).stem))
        selected = pred[pred[:, 4] >= settings['recall_conf']]
        tp = matched(selected)[:, 0]
        for c in range(4):
            totals[c] += int((labels[:, 0] == c).sum())
            mask = selected[:, 5].numpy() == c
            fixed_tp[c] += int(tp[mask].sum())
            fixed_pred[c] += int(mask.sum())
    metrics.process()
    per_class = {}
    indices = list(metrics.box.ap_class_index)
    for c, name in enumerate(NAMES):
        ap = metrics.box.all_ap[indices.index(c)] if c in indices else np.zeros(10)
        per_class[name] = dict(ap50=float(ap[0]), ap50_95=float(ap.mean()), targets=int(totals[c]),
                               tp=int(fixed_tp[c]), fp=int(fixed_pred[c] - fixed_tp[c]), fn=int(totals[c] - fixed_tp[c]),
                               recall=float(fixed_tp[c] / totals[c]) if totals[c] else None,
                               precision=float(fixed_tp[c] / fixed_pred[c]) if fixed_pred[c] else 0.)
    return dict(mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map), per_class=per_class,
                recall_conf=settings['recall_conf'], recall_match_iou=.5,
                recall_definition='Class-aware Ultralytics matching at IoU .50, confidence >= recall_conf')


def run(output, device):
    import cv2
    import torch
    from ultralytics import YOLO
    from tools.benchmark_speed import register_modules, Predictor, measure
    manifest = json.loads((output / 'manifest.json').read_text())
    verify(manifest)
    register_modules()
    settings = manifest['config']['settings']
    torch.set_num_threads(settings['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cuda = torch.device('cpu' if device == 'cpu' else f'cuda:{device}')
    if cuda.type == 'cuda':
        torch.cuda.set_device(cuda)
    images = [cv2.imread(r['image']) for r in manifest['records']]
    if any(im is None for im in images):
        raise ValueError('An image could not be decoded')
    shapes = [list(im.shape[:2]) for im in images]
    configs = manifest['config']['models']
    cases = [dict(id=m['id'], name=m['name'], members=[m['id']], weights=[1.]) for m in configs]
    cases += manifest['config']['ensembles']
    by_id = {m['id']: m for m in configs}
    cached, results = {}, []
    for case in cases:
        print(f"Evaluating {case['id']}", flush=True)
        models = [YOLO(by_id[i]['path']) for i in case['members']]
        for model in models:
            if [model.names[i] for i in range(4)] != NAMES or len(model.names) != 4:
                raise ValueError('Wrong checkpoint class mapping')
        args = SimpleNamespace(imgsz=settings['imgsz'], bench_conf=settings['conf'], bench_iou=settings['nms_iou'],
                               max_det=settings['max_det'], device=device, cpu_threads=settings['cpu_threads'],
                               warmup=settings['warmup'], num_iters=settings['iterations'], allow_shared_gpu=False,
                               pipeline='single' if len(models) == 1 else 'ensemble', fusion='wbf',
                               ensemble_weights=case['weights'], merge_iou=settings['merge_iou'])
        predictors = [Predictor(m, args, False) for m in models]
        if len(models) == 1:
            predictions = []
            for image in images:
                p = predictors[0].predict(image)[0].boxes.data.detach().cpu().tolist()
                predictions.append(p)
            cached[case['id']] = predictions
        else:
            predictions = [fuse([cached[i][k] for i in case['members']], shape, case['weights'], settings)
                           for k, shape in enumerate(shapes)]
            # Confirm cached fusion and the timed live path agree on actual predictions.
            from tools.benchmark_speed import ensemble_inference
            import numpy as np
            b, s, c = ensemble_inference(predictors, images[0], args)
            live = np.column_stack((b, s, c))
            if not np.allclose(live, np.asarray(predictions[0]).reshape(-1, 6), atol=1e-3):
                raise ValueError('Cached WBF and live inference differ')
        write(output / 'predictions' / (case['id'] + '.json'),
              dict(manifest_id=manifest['id'], image_names=[r['name'] for r in manifest['records']],
                   shapes=shapes, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class'], predictions=predictions))
        accuracy = score(predictions, manifest['records'], shapes, settings)
        timing = measure(predictors, images, args, cuda)
        result = dict(**case, manifest_id=manifest['id'], accuracy=accuracy, speed=timing,
                      parameters=[sum(p.numel() for p in m.model.parameters()) for m in models],
                      checkpoint_sha256=[by_id[i]['sha256'] for i in case['members']])
        write(output / 'results' / (case['id'] + '.json'), result)
        results.append(result)
        del models, predictors
        gc.collect()
        if cuda.type == 'cuda':
            torch.cuda.empty_cache()
    verify(manifest)
    summary = dict(status='completed', manifest=manifest, results=results,
                   hardware=dict(gpu=torch.cuda.get_device_name(cuda) if cuda.type == 'cuda' else None,
                                 hostname=platform.node(), slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    write(output / 'summary.json', summary)
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'mAP50_percent', 'mAP50_95_percent', 'Capacitor_AP50_percent',
                         'Capacitor_recall_conf025_percent', 'FP32_mean_ms', 'FP32_FPS'])
        for r in results:
            a, s = r['accuracy'], r['speed']
            writer.writerow([r['name'], 100*a['mAP50'], 100*a['mAP50_95'],
                             100*a['per_class'][NAMES[0]]['ap50'], 100*a['per_class'][NAMES[0]]['recall'],
                             s['wall_prediction']['mean_ms'], s['fps_from_mean_latency']])
    print(f'Completed exact-checkpoint audit: {output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--config', type=Path, default=ROOT / 'configs/champion_audit.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='0')
    a = p.parse_args()
    output = a.output.resolve()
    if a.action == 'prepare':
        prepare(a.config.resolve(), output)
    else:
        run(output, a.device)


if __name__ == '__main__':
    main()
