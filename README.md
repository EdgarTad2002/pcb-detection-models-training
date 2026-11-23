# PCB component detection research

Training, evaluation and paper preparation for four PCB component classes:
Capacitor, Connector, Electrolytic Capacitor and IC.

The project has two separate research tracks:

- **Native 1280:** cleaned annotations, four saved champion models, and fixed
  two-model/four-model WBF ensembles.
- **Paper 640:** original-label RF100-derived data, six methods and three seeds
  under a shared protocol. Published-paper evaluation equivalence remains a
  separate question; see README_INFO17_FINAL.md.

## Start here

- [Current project layout and sync workflow](README_PROJECT_LAYOUT.md)
- [Retained model registry](configs/model_registry.json)
- [Paper workspace](paper/README.md)
- [Exact-checkpoint benchmark protocol](README_CHAMPION_AUDIT.md)
- [Paper-dataset experiment](README_INFO17_FINAL.md)
- [Published-protocol reconciliation](README_INFO17_PROTOCOL.md)
- [Controlled experiment commands](README_CONTROLLED_EXPERIMENTS.md)
- [Clean-only model comparison](README_CLEAN_MAIN_EXPERIMENT.md)
- [Clean/crop comparison](README_CLEAN_CROP_EXPERIMENT.md)
- [Planned research](README_NEXT_EXPERIMENTS.md)

## Evidence and models

Access native champion weights through models/native1280/. On the cluster,
models/paper640/ provides the 18 final method/seed checkpoints. These are links,
not additional checkpoint copies. Original runs and manifests retain their paths.

The primary native benchmark evidence is
results/champion_audit_v1_20261030/summary.json and its per-configuration predictions
and timing samples. The paper640 results are under
controlled_experiments/751837c92ab4/. Keep dataset/protocol labels with every result.
The verified workbook and ten-slide presentation are in paper/reports/champion_audit_verified/.

## Cleanup archive

Approved obsolete model weights and generated data were removed only after
historical configurations, labels/manifests, useful figures and results were
archived and checksums verified. See archive/ receipts and
[the cleanup record](cleanup/README.md). Superseded leaderboards and hardcoded
report generators are historical sources, not current paper results.

Datasets live on the cluster. Some canonical datasets depend on backing folders
through symlinks. Use the reviewed dataset builders and fingerprint checks rather
than deleting apparently old directories by name.

The follow-up [historical presentation/result archival](cleanup/historical_batch_20261031.md) removed superseded working copies after verification; original paths quoted in historical guides may now be archive members.
