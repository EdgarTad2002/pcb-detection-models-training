# Retained model access

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
