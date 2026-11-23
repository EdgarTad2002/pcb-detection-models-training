#!/usr/bin/env python3
"""Apply the approved larger cleanup after verified evidence archival.

Without --execute this performs preflight only. Never deletes review-only assets.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.controlled_experiments import file_hash, read_dataset, split_signature

ACTIVE_DATA = ['pcb-native-res-cleaned-1280', 'pcb-native-res-cleaned-cappaste-1280',
               'pcb-native-mixed-crops-1280', 'pcb-native-mixed-crops-noextraaug-v1',
               'info17-original-fourclass-v1', 'info17-original-mixed-crops-v1']
BANKS = ['capacitor_bank_cleaned', 'capacitor_bank_rectified',
         'capacitor_bank_rectified_640', 'capacitor_bank_rectified_native']
OLD_REPORTS = ['PCB_Detection_Master_Benchmark.csv', 'PCB_Detection_Master_Benchmark.xlsx',
               'PCB_Detection_Model_Weights.csv', 'PCB_Detection_Per_Class_Breakdown.csv',
               'comparison_table.csv', 'comparison_table.xlsx', 'comparison_table.md',
               'presentation/PCB_Detection_Progress.pptx', 'presentation/Professor_Meeting_Presentation.pptx',
               'presentation/Professor_Meeting_Presentation.html', 'presentation/assets',
               'results/comparison_table.csv', 'results/comparison_table.xlsx', 'results/comparison_table.md']


def sha(path):
    return file_hash(str(path.resolve()))


def safe_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] in {'.git', 'archive', 'cleanup', 'ensemble_champions', 'controlled_experiments'}:
        raise ValueError(f'Invalid cleanup path: {relative}')
    target = root / p
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Path escapes project: {relative}')
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f'Symlink cleanup root: {relative}')
    return target


def archive_files(root, paths, destination):
    """Create a verified archive containing exactly the regular/symlink files listed."""
    records = []
    unique = sorted(set(paths), key=str)
    with tarfile.open(destination, 'x:gz', dereference=False) as archive:
        for p in unique:
            if p.is_symlink():
                archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
                records.append({'path': str(p.relative_to(root)), 'symlink': str(p.readlink())})
            elif p.is_file():
                archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
                records.append({'path': str(p.relative_to(root)), 'sha256': sha(p), 'bytes': p.stat().st_size})
    with tarfile.open(destination, 'r:gz') as archive:
        for r in records:
            member = archive.getmember(r['path'])
            if 'symlink' in r:
                if not member.issym() or member.linkname != r['symlink']:
                    raise ValueError('Archived symlink mismatch')
            else:
                stream = archive.extractfile(member)
                h = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
                if h.hexdigest() != r['sha256']:
                    raise ValueError(f"Archive validation failed: {r['path']}")
    return {'archive': str(destination), 'sha256': sha(destination), 'files': records}


def regular_members(p):
    if p.is_file() or p.is_symlink():
        return [p]
    return [q for q in p.rglob('*') if q.is_file() or q.is_symlink()]


def data_fingerprints(root):
    file_hash.cache_clear()
    result = {}
    for name in ACTIVE_DATA:
        path = root / 'datasets' / name / 'data.yaml'
        if path.is_file():
            ds = read_dataset(path)
            result[name] = {s: split_signature(rows) for s, rows in ds['splits'].items()}
    return result


def plan(root, proposal, machine, inventory):
    section = proposal['machines'][machine]
    if root != Path(section['root']).resolve():
        raise ValueError('Proposal belongs to another project root')
    remove_runs = [r for r in section['runs'] if r['action'] == 'archive_then_remove_weights']
    source_rows = [r for r in section['sources'] if r['action'] in {'archive', 'retire'}]
    dataset_rows = [r for r in section['datasets'] if r['action'] == 'archive_manifest_then_remove_generated_dataset']
    removal = {r['path'] for r in remove_runs + dataset_rows + source_rows}
    removal.update(p for p in ['scratch', 'runs/detect', *OLD_REPORTS] if (root / p).exists())
    if machine == 'cluster':
        removal.update(p for p in [*BANKS, 'ensemble_cache'] if (root / p).exists())
    # A directory action subsumes its children. Do not delete a moved child twice.
    removal = sorted(p for p in removal if not any(p.startswith(q + '/') for q in removal if q != p))
    for relative in removal:
        p = safe_path(root, relative)
        if not p.exists():
            raise FileNotFoundError(f'Approved target missing: {p}')
    expected = {r['path']: r for r in section['sources']}
    for row in source_rows:
        p = root / row['path']
        if sha(p) != row['sha256']:
            raise ValueError(f"Source changed since approval: {row['path']}")
    removed_checkpoints = {c['path']: c for r in remove_runs for c in r['checkpoints']}
    for row in remove_runs:
        p = root / row['path']
        actual = {str(q.relative_to(root)) for q in p.rglob('*.pt') if not q.is_symlink()}
        if actual != {c['path'] for c in row['checkpoints']}:
            raise ValueError(f'Checkpoint membership changed: {p}')
        for c in row['checkpoints']:
            if sha(root / c['path']) != c['sha256']:
                raise ValueError(f"Checkpoint changed: {c['path']}")
    # Phase-one inventory also covers root initialization weights and champions.
    checkpoint_rows = inventory['checkpoint_inventory']
    phase_one_removed = {f['path'] for f in inventory['proposed_files']}
    retained = []
    for c in checkpoint_rows:
        p = root / c['path']
        if c['path'] in phase_one_removed:
            continue  # Already explicitly removed and verified in phase one.
        if c['path'] not in removed_checkpoints:
            if not p.is_file() or sha(p) != c['sha256']:
                raise ValueError(f"Retained checkpoint missing or changed: {c['path']}")
            retained.append(c)
    deleted_datasets = {Path(r['path']).name for r in dataset_rows}
    protected_names = proposal['protected_dataset_names']
    for name in protected_names:
        for p in (root / 'datasets' / name).rglob('*'):
            if p.is_symlink():
                target = p.resolve()
                if target.is_relative_to(root / 'datasets') and target.relative_to(root / 'datasets').parts[0] in deleted_datasets:
                    raise ValueError(f'Retained dataset depends on a removal target: {p}')
    evidence, generated_meta = [], []
    for relative in removal:
        p = root / relative
        members = regular_members(p)
        if relative.startswith('datasets/') or relative in BANKS:
            generated_meta.extend(q for q in members if q.suffix.lower() in {'.json', '.yaml', '.yml', '.txt', '.csv', '.md'})
        else:
            # Old unique .pt weights are intentionally not retained in archives.
            evidence.extend(q for q in members if q.suffix.lower() != '.pt' or q.is_symlink())
    # Snapshot current source/config/tests too, for restoration of old hash-locked workflows.
    source_snapshot = [root / r['path'] for r in section['sources'] if (root / r['path']).is_file()]
    return dict(removal=removal, evidence=evidence, generated_meta=generated_meta,
                source_snapshot=source_snapshot, retained=retained,
                removed_checkpoints=list(removed_checkpoints.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal', type=Path, required=True)
    p.add_argument('--machine', choices=['local', 'cluster'], required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    proposal = json.loads(a.proposal.read_text())
    inventory = json.loads(a.inventory.read_text())
    file_hash.cache_clear()
    prepared = plan(ROOT, proposal, a.machine, inventory)
    fingerprints = data_fingerprints(ROOT)
    print(json.dumps({'status': 'preflight_passed', 'machine': a.machine,
                      'working_paths_to_remove': len(prepared['removal']),
                      'old_checkpoints_to_delete': len(prepared['removed_checkpoints']),
                      'retained_checkpoints': len(prepared['retained']),
                      'active_datasets_fingerprinted': list(fingerprints)}, indent=2), flush=True)
    if not a.execute:
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    archive_root = ROOT / 'archive' / f'cleanup_{stamp}_{a.machine}'
    archive_root.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='archiving', machine=a.machine, created_utc=stamp,
                   proposal_sha256=sha(a.proposal), retained_checkpoints=prepared['retained'],
                   intended_removal=prepared['removal'], removed=[], archives=[], dataset_fingerprints_before=fingerprints)
    receipt_path = archive_root / 'receipt.json'
    def save():
        temp = receipt_path.with_suffix('.tmp')
        temp.write_text(json.dumps(receipt, indent=2) + '\n')
        temp.replace(receipt_path)
    save()
    try:
        for name, files in [('source_before_cleanup', prepared['source_snapshot']),
                            ('historical_evidence', prepared['evidence']),
                            ('generated_dataset_metadata', prepared['generated_meta'])]:
            receipt['archives'].append(archive_files(ROOT, files, archive_root / (name + '.tar.gz')))
            save()
        shutil.copyfile(a.proposal, archive_root / 'approved_proposal.json')
        receipt['status'] = 'removing_approved_paths'
        save()
        # Archives were opened and every archived member's checksum verified above.
        for relative in prepared['removal']:
            target = safe_path(ROOT, relative)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            receipt['removed'].append(relative)
            save()
        file_hash.cache_clear()
        for c in prepared['retained']:
            if sha(ROOT / c['path']) != c['sha256']:
                raise ValueError(f"Retained checkpoint changed: {c['path']}")
        after = data_fingerprints(ROOT)
        if after != fingerprints:
            raise ValueError('A retained active dataset changed during cleanup')
        receipt.update(status='completed', dataset_fingerprints_after=after,
                       deleted_checkpoint_logical_bytes=sum(c['bytes'] for c in prepared['removed_checkpoints']))
        save()
        print(json.dumps({'status': 'completed', 'receipt': str(receipt_path),
                          'removed_paths': len(receipt['removed']), 'retained_checkpoints': len(prepared['retained']),
                          'active_datasets_unchanged': True}, indent=2), flush=True)
    except Exception as exc:
        receipt.update(status='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
