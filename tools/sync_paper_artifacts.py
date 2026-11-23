#!/usr/bin/env python3
"""Sync explicit retained artifacts, never historical wildcard checkpoint trees."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = '/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training'


def selections(registry, groups):
    chosen = []
    if 'champions' in groups or 'all' in groups:
        chosen.extend({'path': c['path'], 'sha256': c['sha256']} for c in registry['champions'])
    for run in registry['runs']:
        if run['group'] in groups or 'all' in groups:
            chosen.extend(run['checkpoints'])
    return list({c['path']: c for c in chosen}.values())


def safe_relative(path):
    p = Path(path)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] in ('archive', 'datasets', '.git'):
        raise ValueError('Invalid sync path: ' + path)
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, default=ROOT / 'configs/model_registry.json')
    p.add_argument('--checkpoints', nargs='+', choices=['champions', 'paper640', 'native1280', 'alternates', 'historical_review', 'all'], default=['champions'])
    p.add_argument('--with-run-metadata', action='store_true')
    p.add_argument('--host', default='cluster.ysu.am')
    p.add_argument('--execute', action='store_true', help='Default is a preview with no network access or writes')
    a = p.parse_args()
    registry = json.loads(a.registry.read_text())
    chosen = selections(registry, a.checkpoints)
    commands = []
    common = ['rsync', '-az', '--rsync-path=/mnt/weka/shared-cache/miniforge3/bin/rsync']
    for item in chosen:
        rel = safe_relative(item['path'])
        target = ROOT / rel
        if a.execute:
            target.parent.mkdir(parents=True, exist_ok=True)
        commands.append([*common, f'{a.host}:{REMOTE_ROOT}/{rel}', str(target)])
    directories = [*registry['current_result_directories'], registry['paper_plan']]
    if a.with_run_metadata:
        directories += [r['path'] for r in registry['runs'] if r['group'] != 'historical_review']
    for directory in dict.fromkeys(directories):
        rel = safe_relative(directory)
        target = ROOT / rel
        if a.execute:
            target.mkdir(parents=True, exist_ok=True)
        commands.append([*common, '--ignore-missing-args', '--exclude=*.pt', '--exclude=*.cache',
                         '--exclude=__pycache__/', f'{a.host}:{REMOTE_ROOT}/{rel}/', str(target) + '/'])
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
        if a.execute:
            subprocess.run(cmd, check=True)
    if a.execute:
        for item in chosen:
            path = ROOT / item['path']
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != item['sha256']:
                raise ValueError('Downloaded checkpoint differs from registry: ' + item['path'])
    print(f"{'Synced and hash-verified' if a.execute else 'Preview only:'} {len(chosen)} checkpoints; no archive or dataset directories selected.")


if __name__ == '__main__':
    main()
