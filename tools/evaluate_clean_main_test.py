#!/usr/bin/env python3
"""Freeze and evaluate all 21 clean-main checkpoints on the canonical test split."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.controlled_experiments import NAMES, file_hash, read_dataset, split_signature, verify_plan
from tools.benchmark_speed import assert_completed, sha256


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def collect_cases(bundle, root):
    plan_paths = [Path(p) for p in bundle['new_plans']]
    plans = [json.loads(p.read_text()) for p in plan_paths]
    control_path = Path(bundle['control_plan'])
    control = json.loads(control_path.read_text())
    for plan in [*plans, control]:
        if Path(plan['root']).resolve() != root.resolve():
            raise ValueError('Plan belongs to another project root')
        verify_plan(plan)
    config = plans[0]['config']
    selected = [(plan, job) for plan in plans for job in plan['jobs']]
    for reused in bundle['reused_controls']:
        matches = [j for j in control['jobs'] if j['key'] == reused['job_key']]
        if len(matches) != 1:
            raise ValueError('Reused control must refer to exactly one original job')
        selected.append((control, matches[0]))
    expected = {(m, s) for m in config['studies']['main']['models'] for s in config['seeds']}
    seen, cases = set(), []
    fingerprints = plans[0]['dataset_fingerprints']['clean']
    for plan, job in selected:
        key = (job['model'], job['seed'])
        if key in seen or job['dataset'] != 'clean' or job['epochs'] != 100:
            raise ValueError('Duplicate, non-clean or wrong-budget job in main comparison')
        seen.add(key)
        for field in ('common', 'ultralytics_version'):
            if plan['config'][field] != config[field]:
                raise ValueError(f'Controls differ in {field}')
        if plan['config']['models'][job['model']] != config['models'][job['model']]:
            raise ValueError('Model initializer/loss definitions differ')
        if plan['dataset_fingerprints']['clean'] != fingerprints:
            raise ValueError('Canonical split fingerprints differ across runs')
        checkpoint = root / 'runs' / job['key'] / 'pcb-filtered/weights/best.pt'
        validation = Path(plan['output']) / 'validation' / (job['key'] + '.json')
        assert_completed(checkpoint, validation, job['epochs'])
        result = json.loads(validation.read_text())
        if result['seed'] != job['seed'] or result['epochs'] != 100:
            raise ValueError('Validation evidence does not match the run seed/budget')
        cases.append({'model': job['model'], 'seed': job['seed'], 'original_job_key': job['key'],
                      'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
                      'validation_record': str(validation), 'validation_record_sha256': sha256(validation),
                      'training_recipe': config['models'][job['model']]})
    if seen != expected or len(cases) != 21:
        raise ValueError('Require all seven prespecified models and all three seeds, without duplicates')
    return sorted(cases, key=lambda c: (c['model'], c['seed'])), plans[0], config


def snapshot_fingerprints(path):
    file_hash.cache_clear()
    dataset = read_dataset(path)
    return dataset, {s: split_signature(rs) for s, rs in dataset['splits'].items()}


def prepare(root, output, bundle_path):
    import yaml
    if output.exists():
        raise FileExistsError(f'Test evaluation already prepared: {output}; run it to resume safely')
    bundle = json.loads(bundle_path.read_text())
    cases, plan, config = collect_cases(bundle, root)
    required = config['ultralytics_version']
    if importlib.metadata.version('ultralytics') != required:
        raise ValueError(f'Use ultralytics=={required}')
    source = Path(plan['output']) / 'datasets/clean/data.yaml'
    dataset, fingerprints = snapshot_fingerprints(source)
    if fingerprints != plan['dataset_fingerprints']['clean']:
        raise ValueError('Dataset changed during preparation')
    output.mkdir(parents=True)
    directory = output / 'dataset'
    directory.mkdir()
    for split, records in dataset['splits'].items():
        (directory / (split + '.txt')).write_text(''.join(r['image'] + '\n' for r in records))
    yaml_path = directory / 'data.yaml'
    yaml_path.write_text(yaml.safe_dump({'path': str(directory), 'nc': 4, 'names': NAMES,
        **{s: str(directory / (s + '.txt')) for s in ('train', 'val', 'test')}}))
    common = config['common']
    manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'project_root': str(root),
                'framework_version': required, 'protocol': config['version'], 'cases': cases,
                'dataset_yaml': str(yaml_path), 'dataset_fingerprints': fingerprints,
                'test_image_count': len(dataset['splits']['test']), 'names': NAMES,
                'checkpoint_selection': 'Existing validation-selected best.pt for every prespecified model and seed',
                'settings': {'split': 'test', 'imgsz': common['imgsz'], 'batch': common['batch'],
                             'workers': common['workers'], 'rect': True, 'agnostic_nms': False,
                             'conf': common['eval_conf'], 'iou': common['eval_iou'],
                             'max_det': common['max_det'], 'precision': 'fp32', 'augment': False},
                'source_bundle': str(bundle_path), 'source_bundle_sha256': sha256(bundle_path),
                'source_code_sha256': sha256(__file__),
                'disclosure': 'Historical test scores informed development and labels were corrected; this is not a newly untouched holdout.'}
    from tools.benchmark_speed import CUSTOM_MODULES
    paths = [Path(__file__), root / 'tools/benchmark_speed.py', *[root / (name + '.py') for name in CUSTOM_MODULES]]
    manifest['source_hashes'] = {str(path): sha256(path) for path in paths if path.is_file()}
    write(output / 'manifest.json', manifest)
    print(f"Frozen {len(cases)} checkpoints; {manifest['test_image_count']} canonical test boards. No inference performed.")


def check_result(result, case, manifest):
    expected = {'model': case['model'], 'seed': case['seed'], 'checkpoint_sha256': case['checkpoint_sha256'],
                'eval_split': 'test', 'test_fingerprint': manifest['dataset_fingerprints']['test'],
                'settings': manifest['settings']}
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError('Existing result does not match frozen checkpoint, split or settings')


def summarize(manifest, results):
    if len(results) != 21:
        raise ValueError('Incomplete test comparison')
    groups = {}
    index = {(c['model'], c['seed']): c for c in manifest['cases']}
    seen = set()
    for r in results:
        key = (r['model'], r['seed'])
        if key in seen or key not in index:
            raise ValueError('Duplicate or unexpected model/seed result')
        check_result(r, index[key], manifest)
        seen.add(key)
        groups.setdefault(r['model'], []).append(r)
    if seen != set(index):
        raise ValueError('Missing model/seed results')
    rows = []
    for model, values in groups.items():
        values.sort(key=lambda r: r['seed'])
        if [r['seed'] for r in values] != [0, 1, 2]:
            raise ValueError('Require seeds 0/1/2')
        getters = {'mAP50': lambda r: r['mAP50'], 'mAP50_95': lambda r: r['mAP50_95']}
        for name in NAMES:
            for field in ('per_class_ap50', 'per_class_ap50_95'):
                getters[field + '/' + name] = lambda r, n=name, f=field: r[f][n]
        metrics = {}
        for field, getter in getters.items():
            data = [100 * getter(r) for r in values]
            metrics[field] = {'mean': statistics.mean(data), 'sample_sd': statistics.stdev(data), 'values': data}
        rows.append({'model': model, 'seeds': [0, 1, 2], 'metrics': metrics})
    rows.sort(key=lambda r: r['metrics']['mAP50']['mean'], reverse=True)
    return {'status': 'completed', 'eval_split': 'test', 'test_images': manifest['test_image_count'],
            'successful_runs': 21, 'units': 'percent', 'uncertainty': 'sample standard deviation across seeds, not confidence intervals',
            'settings': manifest['settings'], 'disclosure': manifest['disclosure'], 'summary': rows}


def run(output, device):
    import csv
    import gc
    import torch
    from tools.benchmark_speed import precision_options, register_modules
    manifest = json.loads((output / 'manifest.json').read_text())
    if importlib.metadata.version('ultralytics') != manifest['framework_version']:
        raise ValueError('Framework version differs from frozen evaluation')
    for path, expected in manifest['source_hashes'].items():
        if sha256(path) != expected:
            raise ValueError(f'Evaluation source changed after freezing: {path}')
    _, current = snapshot_fingerprints(manifest['dataset_yaml'])
    if current != manifest['dataset_fingerprints']:
        raise ValueError('Frozen dataset membership or contents changed')
    for case in manifest['cases']:
        if sha256(case['checkpoint']) != case['checkpoint_sha256']:
            raise ValueError('A frozen checkpoint changed')
    register_modules()
    from ultralytics import YOLO
    settings = manifest['settings']
    results = []
    for case in manifest['cases']:
        key = f"{case['model']}_s{case['seed']}"
        target = output / 'per_seed' / (key + '.json')
        if target.exists():
            result = json.loads(target.read_text())
            check_result(result, case, manifest)
            print(f'Reusing completed test evaluation: {key}', flush=True)
        else:
            model = YOLO(case['checkpoint'])
            names = [model.names[i] for i in sorted(model.names)] if isinstance(model.names, dict) else model.names
            if names != manifest['names']:
                raise ValueError('Checkpoint class mapping differs from test dataset')
            metrics = model.val(data=manifest['dataset_yaml'], split='test', imgsz=settings['imgsz'],
                    batch=settings['batch'], conf=settings['conf'], iou=settings['iou'],
                    max_det=settings['max_det'], **precision_options(False), device=device,
                    workers=settings['workers'], rect=settings['rect'], agnostic_nms=settings['agnostic_nms'],
                    augment=False, plots=False, save_json=False,
                    project=str(output / 'evaluation_artifacts'), name=key, exist_ok=True)
            indices = metrics.box.ap_class_index
            result = {'model': case['model'], 'seed': case['seed'], 'weights': case['checkpoint'],
                      'checkpoint_sha256': case['checkpoint_sha256'], 'eval_split': 'test',
                      'test_fingerprint': manifest['dataset_fingerprints']['test'], 'settings': settings,
                      'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map),
                      'precision': float(metrics.box.p.mean()), 'recall': float(metrics.box.r.mean()),
                      'per_class_ap50': {metrics.names[int(i)]: float(ap) for i, ap in zip(indices, metrics.box.ap50)},
                      'per_class_ap50_95': {metrics.names[int(i)]: float(ap) for i, ap in zip(indices, metrics.box.ap)},
                      'completed_utc': datetime.now(timezone.utc).isoformat()}
            if sha256(case['checkpoint']) != case['checkpoint_sha256']:
                raise ValueError('Checkpoint changed during evaluation')
            check_result(result, case, manifest)
            write(target, result)
            del model, metrics
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Completed {key}: test AP50={result['mAP50']*100:.2f}%", flush=True)
        results.append(result)
    _, after = snapshot_fingerprints(manifest['dataset_yaml'])
    if after != manifest['dataset_fingerprints']:
        raise ValueError('Dataset changed during evaluation; results require investigation')
    summary = summarize(manifest, results)
    write(output / 'summary.json', summary)
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'AP50_mean_percent', 'AP50_sample_sd', 'AP50_95_mean_percent', 'AP50_95_sample_sd',
                         'Capacitor_AP50_mean_percent', 'Capacitor_AP50_sample_sd', 'Capacitor_AP50_95_mean_percent', 'Capacitor_AP50_95_sample_sd'])
        for row in summary['summary']:
            m = row['metrics']
            writer.writerow([row['model'], *[m[k][f] for k in ('mAP50','mAP50_95','per_class_ap50/Capacitor','per_class_ap50_95/Capacitor') for f in ('mean','sample_sd')]])
    print('Completed all 21 prespecified test evaluations. Validation/speed results were not modified.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'run'))
    p.add_argument('--project-root', type=Path, default=ROOT)
    p.add_argument('--bundle', type=Path, default=Path('controlled_experiments/clean_main_noextraaug_v1_bundle.json'))
    p.add_argument('--output', type=Path, default=Path('results/clean_main_test_v1'))
    p.add_argument('--device', default='0')
    args = p.parse_args()
    root = args.project_root.resolve()
    output = args.output.resolve() if args.output.is_absolute() else root / args.output
    if args.action == 'prepare':
        bundle = args.bundle if args.bundle.is_absolute() else root / args.bundle
        prepare(root, output, bundle)
    else:
        run(output, args.device)


if __name__ == '__main__':
    main()
