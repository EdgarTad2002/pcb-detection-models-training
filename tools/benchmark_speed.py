#!/usr/bin/env python3
"""Benchmark real-board batch-one prediction; write standalone, immutable speed records.

Default selection: completed checkpoints in the clean-main bundle. Explicit
--weights and --plan selections are also supported. Accuracy evaluation is opt-in
and defaults to validation. Historical accuracy JSONs are never updated.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
CUSTOM_MODULES = ('luma_modules', 'retinex_stem', 'weber_stem', 'icm_module',
                  'physics_spectral_yolo26', 'superyolo_pcb', 'squared_dice_loss',
                  'agwd_ota_loss', 'igr_stem', 'fg_csfa_neck', 'aca_supcon')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + '\n')


def checkpoint_id(path, root):
    path = Path(path).resolve()
    try:
        relative = path.relative_to(root.resolve())
    except ValueError:
        relative = path
    label = re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(relative.with_suffix(''))).strip('_')
    return label[-130:] + '_' + hashlib.sha256(str(path).encode()).hexdigest()[:10]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=ROOT)
    p.add_argument('--weights', type=Path, nargs='+')
    p.add_argument('--plan', type=Path, nargs='+')
    p.add_argument('--bundle', type=Path, help='Defaults to the current clean-main bundle')
    p.add_argument('--output-dir', '--results-dir', dest='output_dir', type=Path,
                   help='Fresh standalone benchmark directory; existing directories are rejected')
    p.add_argument('--images-dir', '--test-images-dir', dest='images_dir', type=Path)
    p.add_argument('--data', type=Path, default=Path('datasets/pcb-native-res-cleaned-1280/data.yaml'))
    p.add_argument('--input-split', choices=('val', 'test'), default='val')
    p.add_argument('--synthetic', action='store_true', help='Explicit synthetic-input diagnostic, not real-board speed')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--device', default='0')
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--num-iters', type=int, default=100)
    p.add_argument('--max-samples', type=int, default=80)
    p.add_argument('--max-det', type=int, default=1000)
    p.add_argument('--bench-conf', type=float, default=.10)
    p.add_argument('--bench-iou', type=float, default=.50, help='Prediction NMS IoU, where the detector uses NMS')
    p.add_argument('--precision', nargs='+', choices=('fp32', 'fp16'), default=['fp32', 'fp16'])
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--profile-flops', action='store_true', help='Optional THOP estimate; unsupported FLOPs are null')
    p.add_argument('--allow-shared-gpu', action='store_true', help='Diagnostic only: permit detected sharing or unverified isolation')
    p.add_argument('--pipeline', choices=('single', 'crop', 'ensemble'), default='single')
    p.add_argument('--crop-weights', type=Path)
    p.add_argument('--slice-size', type=int, default=640)
    p.add_argument('--overlap', type=float, default=.2)
    p.add_argument('--fusion', choices=('nms', 'wbf'), default='nms')
    p.add_argument('--merge-iou', type=float, default=.5)
    p.add_argument('--ios-thresh', type=float, default=.65)
    p.add_argument('--ensemble-weights', type=float, nargs='+', help='Positive WBF member weights; default equal')
    p.add_argument('--re-evaluate-map', action='store_true')
    p.add_argument('--eval-split', choices=('val', 'test'), default='val')
    p.add_argument('--eval-conf', type=float, default=.001)
    p.add_argument('--eval-iou', type=float, default=.5, help='Accuracy prediction NMS IoU; not the AP matching threshold')
    p.add_argument('--update-results', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.update_results:
        p.error('--update-results was removed: benchmark outputs never overwrite accuracy records')
    if args.imgsz < 32 or args.imgsz % 32 or args.num_iters < 1 or args.warmup < 0 or args.max_samples < 1 or args.cpu_threads < 1 or args.max_det < 1:
        p.error('Require positive counts, nonnegative warmup and imgsz divisible by 32')
    if not all(0 < x <= 1 for x in (args.bench_conf, args.bench_iou, args.eval_conf, args.eval_iou, args.merge_iou, args.ios_thresh)) or not 0 <= args.overlap < 1 or args.slice_size < 1:
        p.error('Invalid confidence, IoU, overlap or slice size')
    if sum(bool(x) for x in (args.weights, args.plan, args.bundle)) > 1:
        p.error('Choose one of --weights, --plan or --bundle')
    if args.pipeline != 'single' and not args.weights:
        p.error('Crop/ensemble pipelines require explicit --weights')
    if args.pipeline == 'crop' and (len(args.weights) != 1 or not args.crop_weights):
        p.error('Crop pipeline requires one global --weights and --crop-weights')
    if args.pipeline == 'ensemble' and len(args.weights) < 2:
        p.error('Ensemble requires at least two checkpoints')
    if args.pipeline != 'crop' and args.crop_weights:
        p.error('--crop-weights requires --pipeline crop')
    if args.ensemble_weights and (args.pipeline != 'ensemble' or args.fusion != 'wbf' or len(args.ensemble_weights) != len(args.weights) or any(w <= 0 for w in args.ensemble_weights)):
        p.error('--ensemble-weights requires one positive weight per WBF ensemble member')
    if args.re_evaluate_map and (args.pipeline != 'single' or args.synthetic):
        p.error('Accuracy re-evaluation supports real-input single detectors only; use the pipeline evaluator for crop/ensemble AP')
    return args


def resolve(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def select_checkpoints(args):
    root = args.project_root.resolve()
    if args.weights:
        paths = [resolve(root, p) for p in args.weights]
        if args.crop_weights:
            paths.append(resolve(root, args.crop_weights))
    else:
        from tools.controlled_experiments import verify_plan
        if args.plan:
            plan_paths = [resolve(root, p) for p in args.plan]
            reuse = []
        else:
            bundle = resolve(root, args.bundle or 'controlled_experiments/clean_main_noextraaug_v1_bundle.json')
            info = json.loads(bundle.read_text())
            plan_paths = [resolve(root, p) for p in info['new_plans']]
            # Verify the original plan too, including its frozen lists and split hashes.
            original = json.loads(resolve(root, info['control_plan']).read_text())
            if Path(original['root']).resolve() != root:
                raise ValueError('Reused controls belong to a different project root')
            verify_plan(original)
            reuse = info['reused_controls']
        paths = []
        for path in plan_paths:
            plan = json.loads(path.read_text())
            if Path(plan['root']).resolve() != root:
                raise ValueError(f'Plan belongs to a different root: {path}')
            verify_plan(plan)
            for job in plan['jobs']:
                checkpoint = root / 'runs' / job['key'] / 'pcb-filtered/weights/best.pt'
                assert_completed(checkpoint, Path(plan['output']) / 'validation' / (job['key'] + '.json'), job['epochs'])
                paths.append(checkpoint)
        for case in reuse:
            checkpoint = Path(case['weights'])
            assert_completed(checkpoint, Path(case['validation_result']), 100)
            paths.append(checkpoint)
        import importlib.metadata
        checked_plans = [json.loads(path.read_text()) for path in plan_paths]
        if not args.plan:
            checked_plans.append(original)
        for plan in checked_plans:
            required = plan['config']['ultralytics_version']
            if importlib.metadata.version('ultralytics') != required:
                raise ValueError(f'Plan requires ultralytics=={required}')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.pipeline == 'single':
        paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError('No checkpoints selected')
    return paths


def assert_completed(checkpoint, result_path, epochs):
    if not checkpoint.is_file() or not result_path.is_file():
        raise ValueError(f'Run is not ready for final benchmarking: {checkpoint}')
    curve = checkpoint.parent.parent / 'results.csv'
    with curve.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != epochs or int(float(rows[-1]['epoch'])) != epochs:
        raise ValueError(f'Training has not reached its planned epochs: {checkpoint}')
    result = json.loads(result_path.read_text())
    if result.get('eval_split') != 'val' or Path(result['weights']).resolve() != checkpoint.resolve():
        raise ValueError(f'Completion result does not match checkpoint/validation split: {result_path}')


def load_inputs(args):
    """Cache decoded real BGR images in RAM; never silently substitute noise."""
    if args.synthetic:
        if args.images_dir:
            raise ValueError('Do not combine --synthetic with --images-dir')
        rng = np.random.default_rng(0)
        images = [rng.integers(0, 256, (args.imgsz, args.imgsz, 3), dtype=np.uint8)]
        return images, {'kind': 'synthetic_diagnostic', 'seed': 0, 'source_images': []}
    if args.images_dir:
        directory = resolve(args.project_root, args.images_dir)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths = sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        from tools.controlled_experiments import read_dataset
        data = read_dataset(resolve(args.project_root, args.data))
        paths = [Path(r['image']) for r in data['splits'][args.input_split]]
    if not paths:
        raise ValueError('No real benchmark images found')
    images, records = [], []
    for path in paths[:args.max_samples]:
        with Image.open(path) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))[:, :, ::-1].copy()
        images.append(image)
        records.append({'path': str(path), 'sha256': sha256(path), 'shape_hwc': list(image.shape)})
    return images, {'kind': 'real_decoded_boards', 'source_images': records,
                    'input_split': args.input_split if not args.images_dir else 'explicit_directory',
                    'unique_cached_images': len(images)}


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Invalid timing samples')
    return {'count': len(a), 'mean_ms': float(a.mean()), 'median_ms': float(np.median(a)),
            'p90_ms': float(np.percentile(a, 90)), 'p95_ms': float(np.percentile(a, 95)),
            'std_ms': float(a.std(ddof=1)) if len(a) > 1 else 0., 'samples_ms': a.tolist()}


def gpu_isolation(device, allow_shared=False):
    if device.type != 'cuda':
        return {'status': 'not_applicable_cpu'}
    props = torch.cuda.get_device_properties(device)
    uuid = getattr(props, 'uuid', None)
    if uuid is not None:
        uuid = str(uuid)
    # PyTorch exposes the bare UUID, while nvidia-smi prefixes physical GPUs
    # with GPU-. Preserve MIG identifiers rather than equating them to parents.
    def uuid_key(value):
        return str(value).strip().lower().removeprefix('gpu-')
    try:
        hardware = subprocess.run(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'],
                                  capture_output=True, text=True, check=True)
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        if not allow_shared:
            raise RuntimeError('Cannot check GPU isolation using nvidia-smi') from exc
        return {'status': 'unverified', 'error': str(exc), 'sharing_override': True}
    others = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) >= 3 and uuid and uuid_key(row[0]) == uuid_key(uuid) and int(row[1]) != os.getpid():
            others.append({'pid': int(row[1]), 'process': row[2].strip(), 'memory_mib': row[3].strip() if len(row) > 3 else None})
    drivers = {uuid_key(row[0]): (row[0].strip(), row[1].strip())
               for row in csv.reader(hardware.stdout.splitlines()) if len(row) >= 2}
    selected = drivers.get(uuid_key(uuid)) if uuid else None
    verified = selected is not None
    if (others or not verified) and not allow_shared:
        raise RuntimeError(f'GPU isolation is not established: uuid={uuid}, other_compute_processes={others}. Use a dedicated GPU.')
    return {'status': 'other_processes_present' if others else ('no_other_compute_processes_observed' if verified else 'unverified'),
            'uuid': selected[0] if selected else uuid, 'torch_uuid': uuid,
            'driver_version': selected[1] if selected else None,
            'other_compute_processes': others, 'sharing_override': allow_shared}


def register_modules():
    unavailable = {}
    for name in CUSTOM_MODULES:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            unavailable[name] = str(exc)
    return unavailable


class Predictor:
    """Common prediction settings plus accumulation of backend-reported forward timings."""
    def __init__(self, model, args, half):
        self.model, self.args, self.half = model, args, half
        self.precision_options = precision_options(half)
        self.forward_ms = 0.
        self.calls = 0

    def predict(self, image, **overrides):
        options = dict(overrides)
        options.pop('half', None)
        options.pop('quantize', None)
        options.update(imgsz=self.args.imgsz, conf=self.args.bench_conf, iou=self.args.bench_iou,
                       max_det=self.args.max_det, device=self.args.device,
                       verbose=False, batch=1, rect=False, save=False, augment=False)
        options.update(self.precision_options)
        result = self.model.predict(image, **options)
        for r in result:
            self.forward_ms += float(r.speed['inference'])
        self.calls += 1
        return result

    def reset(self):
        self.forward_ms = 0.
        self.calls = 0

    def check_precision(self):
        backend = self.model.predictor.model
        if bool(backend.fp16) != self.half:
            raise RuntimeError('Backend precision differs from requested precision')
        requested = torch.device('cpu' if self.args.device == 'cpu' else f'cuda:{self.args.device}')
        actual = torch.device(backend.device)
        if actual != requested:
            raise RuntimeError(f'Backend device {actual} differs from requested device {requested}')
        return {'fp16': bool(backend.fp16), 'device': str(backend.device),
                'end2end_detector': bool(getattr(backend, 'end2end', False))}


def ensemble_inference(predictors, image, args):
    import torchvision
    boxes, scores, classes = [], [], []
    height, width = image.shape[:2]
    for predictor in predictors:
        res = predictor.predict(image)[0]
        boxes.append(res.boxes.xyxy.detach().cpu())
        scores.append(res.boxes.conf.detach().cpu())
        classes.append(res.boxes.cls.detach().cpu().long())
    if args.fusion == 'wbf':
        from ensemble_boxes import weighted_boxes_fusion
        scale = torch.tensor([width, height, width, height])
        result = weighted_boxes_fusion([(b / scale).clamp(0, 1).tolist() for b in boxes],
                                      [s.tolist() for s in scores], [c.tolist() for c in classes],
                                      weights=args.ensemble_weights, iou_thr=args.merge_iou,
                                      skip_box_thr=args.bench_conf)
        fused_boxes, fused_scores, fused_classes = result
        fused_boxes = fused_boxes * np.array([width, height, width, height])
        return tuple(x[:args.max_det] for x in (fused_boxes, fused_scores, fused_classes))
    boxes, scores, classes = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
    keep = torchvision.ops.batched_nms(boxes, scores, classes, args.merge_iou)[:args.max_det]
    return boxes[keep], scores[keep], classes[keep]


def run_pipeline(predictors, image, args):
    if args.pipeline == 'single':
        return predictors[0].predict(image)
    if args.pipeline == 'ensemble':
        return ensemble_inference(predictors, image, args)
    from tools.eval_dual_branch_pipeline import run_dual_branch_inference
    return run_dual_branch_inference(predictors[0], predictors[1], image,
               slice_size=args.slice_size, overlap=args.overlap, conf_thresh=args.bench_conf,
               merge_iou=args.merge_iou, ios_thresh=args.ios_thresh, max_det=args.max_det,
               fusion_method=args.fusion, device=args.device, imgsz=args.imgsz,
               prediction_iou=args.bench_iou)


def measure(predictors, images, args, device):
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    isolation_before = gpu_isolation(device, args.allow_shared_gpu)
    # Backend initialization can reset Torch's CPU thread count. Initialize every
    # branch first, restore the requested threads, then warm up under that setting.
    run_pipeline(predictors, images[0], args)
    torch.set_num_threads(args.cpu_threads)
    for index in range(args.warmup):
        run_pipeline(predictors, images[index % len(images)], args)
    sync()
    precision = [p.check_precision() for p in predictors]
    wall, backend, calls, cuda_intervals = [], [], [], []
    for index in range(args.num_iters):
        for p in predictors:
            p.reset()
        image = images[index % len(images)]
        sync()
        if device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(device))
        t0 = time.perf_counter()
        run_pipeline(predictors, image, args)
        if device.type == 'cuda':
            end.record(torch.cuda.current_stream(device))
        sync()
        wall.append((time.perf_counter() - t0) * 1000)
        if device.type == 'cuda':
            cuda_intervals.append(float(start.elapsed_time(end)))
        backend.append(sum(p.forward_ms for p in predictors))
        calls.append(sum(p.calls for p in predictors))
    isolation_after = gpu_isolation(device, args.allow_shared_gpu)
    summary = stats(wall)
    return {'wall_prediction': summary, 'fps_from_mean_latency': 1000 / summary['mean_ms'],
            'fps_from_median_latency': 1000 / summary['median_ms'],
            'backend_reported_forward_sum': stats(backend),
            'cuda_pipeline_interval': stats(cuda_intervals) if cuda_intervals else None,
            'cuda_interval_note': 'Pipeline stream interval may include host scheduling gaps; not pure GPU compute time.',
            'detector_calls_per_board': calls, 'backend_precision': precision,
            'torch_threads_during_timing': torch.get_num_threads(),
            'isolation_before': isolation_before, 'isolation_after': isolation_after,
            'timing_scope': 'Cached decoded board through prediction and pipeline fusion; excludes disk decoding, model loading, setup and warmup.'}


def evaluate_accuracy(model, args, output):
    from tools.controlled_experiments import read_dataset, split_signature, NAMES
    dataset = read_dataset(resolve(args.project_root, args.data))
    names = model.names
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else names
    if names != NAMES:
        raise ValueError('Accuracy class mapping does not match canonical four-class dataset')
    metrics = model.val(data=dataset['yaml'], split=args.eval_split, imgsz=args.imgsz, batch=1,
                        conf=args.eval_conf, iou=args.eval_iou, max_det=args.max_det,
                        device=args.device, **precision_options(False), plots=False, save_json=False,
                        project=str(output), name='accuracy_artifacts', exist_ok=False)
    return {'split': args.eval_split, 'data_yaml': dataset['yaml'],
            'split_fingerprint': split_signature(dataset['splits'][args.eval_split]),
            'precision': 'fp32', 'conf': args.eval_conf, 'prediction_nms_iou': args.eval_iou,
            'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map),
            'per_class_ap50': {metrics.names[int(i)]: float(ap) for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap50)},
            'per_class_ap50_95': {metrics.names[int(i)]: float(ap) for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap)}}


def complexity(model, args):
    result = {'parameters': sum(p.numel() for p in model.model.parameters()),
              'gflops_estimate': None, 'flops_status': 'not_requested'}
    if args.profile_flops:
        from ultralytics.utils.torch_utils import get_flops
        try:
            value = get_flops(model.model, args.imgsz)
            if value and np.isfinite(value) and value > 0:
                result.update(gflops_estimate=float(value), flops_status='thop_estimate_custom_functional_ops_may_be_unaccounted')
            else:
                result['flops_status'] = 'unavailable_or_unsupported'
        except Exception as exc:
            result['flops_status'] = f'unsupported: {exc}'
    return result


def precision_options(half):
    from ultralytics.cfg import DEFAULT_CFG_DICT
    # New versions renamed half to quantize. Passing half on every iteration
    # prints deprecation warnings inside the measured prediction call.
    return {'quantize': 16 if half else None} if 'quantize' in DEFAULT_CFG_DICT else {'half': half}


def validate_pipeline_classes(models, pipeline):
    def names(model):
        return [model.names[i] for i in sorted(model.names)] if isinstance(model.names, dict) else model.names
    if pipeline != 'single' and any(names(m) != names(models[0]) for m in models):
        raise ValueError('Pipeline members have different class mappings')
    if pipeline == 'crop':
        from tools.controlled_experiments import NAMES
        if names(models[0]) != NAMES:
            raise ValueError('The capacitor crop pipeline requires the canonical four-class mapping')


def measure_precision(paths, images, args, device, half):
    from ultralytics import YOLO
    models, predictors = [], []
    try:
        for path in paths:
            model = YOLO(str(path))
            models.append(model)
            predictors.append(Predictor(model, args, half))
        return measure(predictors, images, args, device)
    finally:
        predictors.clear()
        models.clear()
        # The loop variable may hold the final model even after clearing the list.
        model = None
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def main(argv=None):
    args = parse_args(argv)
    args.project_root = args.project_root.resolve()
    device = torch.device('cpu' if args.device == 'cpu' else (f'cuda:{args.device}' if args.device.isdigit() else args.device))
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use cpu or a single CUDA device')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; use --device cpu explicitly for CPU checks')
        device = torch.device('cuda', device.index or 0)
        torch.cuda.set_device(device)
        args.device = str(device.index)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = select_checkpoints(args)
    images, source = load_inputs(args)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = resolve(args.project_root, args.output_dir or Path('results/speed_benchmarks') / stamp)
    output.mkdir(parents=True, exist_ok=False)
    unavailable = register_modules()
    from ultralytics import YOLO, __version__
    metadata = {'schema_version': 2, 'created_utc': stamp, 'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'input': source, 'torch': str(torch.__version__), 'ultralytics': __version__,
                'python': platform.python_version(), 'platform': platform.platform(),
                'hostname': platform.node(),
                'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                'cpu_threads': args.cpu_threads, 'cuda_runtime': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'gpu_properties': str(torch.cuda.get_device_properties(device)) if device.type == 'cuda' else None,
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'unavailable_custom_imports': unavailable,
                'timed_iterations_per_precision': args.num_iters,
                'timed_input_indices': [i % len(images) for i in range(args.num_iters)],
                'preprocessing': 'Ultralytics batch=1, rect=False, square letterbox to imgsz; original decoded BGR boards cached in RAM',
                'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
                'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
                'records': [], 'failures': []}
    metadata['code_sha256'] = {str(Path(__file__).resolve()): sha256(__file__)}
    if args.pipeline == 'crop':
        helper = ROOT / 'tools/eval_dual_branch_pipeline.py'
        metadata['code_sha256'][str(helper)] = sha256(helper)
    import importlib.metadata
    metadata['dependency_versions'] = {}
    for package in ('numpy', 'Pillow', 'torchvision', 'ensemble-boxes', 'ultralytics-thop'):
        try:
            metadata['dependency_versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata['dependency_versions'][package] = None
    save(output / 'benchmark.json', metadata)
    try:
        metadata['preflight_gpu_isolation'] = gpu_isolation(device, args.allow_shared_gpu)
    except Exception as exc:
        metadata['failures'].append({'stage': 'gpu_isolation_preflight', 'error': str(exc)})
        save(output / 'benchmark.json', metadata)
        print(f'GPU isolation preflight failed: {exc}', flush=True)
        return 1
    save(output / 'benchmark.json', metadata)
    groups = [[p] for p in paths] if args.pipeline == 'single' else [paths]
    for group in groups:
        identity = '__'.join(checkpoint_id(p, args.project_root) for p in group)
        identity = identity[:180] + '_' + hashlib.sha256(identity.encode()).hexdigest()[:10]
        record = {'id': identity, 'pipeline': args.pipeline,
                  'checkpoints': [{'path': str(p), 'sha256': sha256(p)} for p in group], 'precision': {}}
        destination = output / identity
        destination.mkdir()
        info = model = None
        try:
            info = [YOLO(str(path)) for path in group]
            validate_pipeline_classes(info, args.pipeline)
            record['complexity_per_checkpoint'] = [complexity(model, args) for model in info]
            record['parameters_total'] = sum(x['parameters'] for x in record['complexity_per_checkpoint'])
            record['flops_scope'] = 'Per detector forward at square imgsz; excludes NMS/fusion. Crop board cost depends on recorded detector call counts.'
            del info
            for precision in dict.fromkeys(args.precision):
                if precision == 'fp16' and device.type == 'cpu':
                    record['precision'][precision] = {'status': 'unsupported_on_cpu'}
                    continue
                record['precision'][precision] = measure_precision(group, images, args, device, precision == 'fp16')
            if args.re_evaluate_map:
                model = YOLO(str(group[0]))
                record['accuracy'] = evaluate_accuracy(model, args, destination)
                del model
            for item in record['checkpoints']:
                if sha256(item['path']) != item['sha256']:
                    raise ValueError('Checkpoint changed during benchmarking')
            record['status'] = 'completed'
        except Exception as exc:
            record['status'] = 'failed'
            record['error'] = str(exc)
            metadata['failures'].append({'id': identity, 'error': str(exc)})
        finally:
            info = model = None
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        metadata['records'].append(record)
        save(destination / 'record.json', record)
        save(output / 'benchmark.json', metadata)
        print(identity, record['status'], flush=True)
    fields = ['id', 'pipeline', 'precision', 'parameters', 'mean_ms', 'median_ms', 'p95_ms', 'fps', 'iterations']
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in metadata['records']:
            if record['status'] != 'completed':
                continue
            for precision, timing in record['precision'].items():
                if 'wall_prediction' in timing:
                    s = timing['wall_prediction']
                    writer.writerow(dict(id=record['id'], pipeline=record['pipeline'], precision=precision,
                        parameters=record.get('parameters_total'), mean_ms=s['mean_ms'], median_ms=s['median_ms'],
                        p95_ms=s['p95_ms'], fps=timing['fps_from_mean_latency'], iterations=s['count']))
    print(f'Saved standalone benchmark: {output}')
    return 1 if metadata['failures'] else 0


if __name__ == '__main__':
    sys.exit(main())
