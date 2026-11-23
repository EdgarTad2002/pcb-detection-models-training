# Paper workspace

- `native1280/results/`: exact four-champion and two-ensemble evaluation, raw
  predictions, timing samples and hashes. The cleaned test split has 44 boards.
- `paper640/results/`: final original-label dataset experiment, 18 model/seed
  runs. Read README_INFO17_FINAL.md before comparing with the published paper.
- `figures/`, `tables/`, `reports/`, `manuscript/`, `references/`: destinations for
  new paper outputs. Historical generated reports live in archive/ and must not
  be presented as current results.

Use JSON evaluation records as primary evidence. The verified workbook and ten-slide presentation are in
reports/champion_audit_verified/ with an export_provenance.json record. These experiments use different
datasets/protocols and should not be combined in one unlabeled leaderboard.

All links preserve the existing frozen run and dataset paths. The model registry
is configs/model_registry.json. Baselines, parameter studies and negative results
remain part of the research evidence even when they are not selected champions.

For the additional original-label evaluation audit, see `README_INFO17_PROTOCOL.md` at the project root. Public-source evaluator views remain diagnostic; published-table equivalence is unverified.

The comprehensive 30-slide research presentation is in `reports/master_research_presentation_20261031/PCB_Research_Master_Presentation_2026-10-31_Final.pptx`. It includes per-class AP/P/R, overall pooled P/R/F1, both native WBF ensembles, verified model counts and isolated native-1280/paper-640 speed measurements. Published scores remain contextual pending exact protocol equivalence.
