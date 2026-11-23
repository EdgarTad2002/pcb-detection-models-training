#!/usr/bin/env python3
"""Create paper-facing links without moving hash-locked runs or duplicating weights."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link_if_present(destination, target):
    source = ROOT / target
    if not source.exists():
        return False
    destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError('Existing workspace link belongs to another artifact: ' + str(destination))
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())
    return True


def main():
    registry = json.loads((ROOT / 'configs/model_registry.json').read_text())
    native, paper = 0, 0
    for model in registry['champions']:
        native += link_if_present('models/native1280/' + model['id'] + '.pt', model['path'])
    for run in registry['runs']:
        if run['group'] != 'paper640':
            continue
        name = Path(run['path']).name
        seed = name.rsplit('_s', 1)[1]
        if '_main_' in name:
            method = name.split('_main_', 1)[1].rsplit('_paper_s', 1)[0]
        else:
            method = 'yolo26s_crop_specialist'
        for checkpoint in run['checkpoints']:
            paper += link_if_present(f'models/paper640/{method}/seed{seed}.pt', checkpoint['path'])
    for folder in ['paper/figures', 'paper/tables', 'paper/reports', 'paper/manuscript',
                   'paper/references', 'models/paper640', 'paper/native1280', 'paper/paper640']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    link_if_present('paper/native1280/results', 'results/champion_audit_v1_20261030')
    link_if_present('paper/paper640/results', registry['paper_plan'])
    (ROOT / 'models/README.md').write_text('''# Retained model access

`native1280/` links to the four exact champion checkpoints. `paper640/` links to
the six final paper-dataset methods and three seeds when those checkpoints are
present on this computer. The links do not duplicate weights or change run paths.

The authoritative list and hashes are in `configs/model_registry.json`. The
cluster retains all 18 paper640 checkpoints. To download them locally, run:

```bash
python3 tools/sync_paper_artifacts.py --checkpoints paper640 --execute
python3 tools/organize_paper_workspace.py
```

Historical-review weights are not included in default downloads.
''')
    (ROOT / 'paper/README.md').write_text('''# Paper workspace

- `native1280/results/`: exact four-champion and two-ensemble evaluation, raw
  predictions, timing samples and hashes. The cleaned test split has 44 boards.
- `paper640/results/`: final original-label dataset experiment, 18 model/seed
  runs. Read README_INFO17_FINAL.md before comparing with the published paper.
- `figures/`, `tables/`, `reports/`, `manuscript/`, `references/`: destinations for
  new paper outputs. Historical generated reports live in archive/ and must not
  be presented as current results.

Use JSON evaluation records as primary evidence. Reports in paper/reports/ are derived outputs. Check their
export_provenance.json records before using them for publication. These experiments use different
datasets/protocols and should not be combined in one unlabeled leaderboard.

All links preserve the existing frozen run and dataset paths. The model registry
is configs/model_registry.json. Baselines, parameter studies and negative results
remain part of the research evidence even when they are not selected champions.
''')
    print(json.dumps({'native1280_model_links': native, 'paper640_model_links': paper,
                      'weights_duplicated': False, 'existing_run_paths_changed': False}))


if __name__ == '__main__':
    main()
