# Experimental Setup and Results - draft for discussion

## Experimental scope and dataset
We investigate four-class PCB component detection using the available
Roboflow-100 printed-circuit-board version-3 exports. The controlled
original-label experiment contains 543 training, 77 validation and 41 test image
exports. The original relevant categories are mapped to Capacitor, Connector,
Electrolytic Capacitor and IC without cleaning their labels or adding offline
copy-paste samples. The test annotations contain 1,920 capacitors, 374
connectors, 18 electrolytic capacitors and 574 ICs. Existing labels and
polygon-to-tight-box conversion are documented in the dataset manifest.
These are image exports, not necessarily independent physical boards. This
RF100 source has informed earlier development and is not an independently
unseen external generalization dataset. A filename audit of the original-label
manifest also finds three cross-split view aliases (ArduinoMega top, DuetWIFI
bottom and DuetWIFI top), mirroring the view groups flagged in the native-data
review. This is a grouping diagnostic, not a new pixel-level verification of
every original-label export. Results on these inherited splits support
benchmark-specific comparisons and do not certify board-disjoint generalization.

## Implementation and training
The completed study includes YOLOv8s, YOLOv12s, YOLO26s, AGWD plus Squared-Dice,
PhysOT-YOLO26s, and a crop-trained AGWD-plus-Dice variant. Independent runs use
seeds 0, 1 and 2 and general pretrained initializers, with checkpoints selected
by validation performance. Full-board methods train for 100 epochs; the
mixed-crop variant trains for 50 epochs under an approximately matched update
budget. All use 640-pixel input, batch size 16, nominal batch size 16, SGD,
initial learning rate 0.01, momentum 0.937, weight decay 0.0005 and a scaled
warmup. Training enables the recorded standard online augmentations; offline
copy-paste and mixup are disabled. The crop variant mixes full boards with
training-only 320/480-pixel windows; its close-mosaic schedule is scaled to five
epochs rather than ten. This is a comparison of complete recipes, not an
isolated causal test of every augmentation or architecture component.

The AGWD-plus-Dice recipe uses Dice weight 0.50, smoothing 0.01, equal stage
weights, and AGWD pixel constants kappa=6.4 and tau=144 at input size 640.
PhysOT combines AGWD with IGR, FG-CSFA and ACA-SupCon. Exact commands, checkpoint
hashes, dataset hashes and dependency versions are retained in the frozen
experiment manifests.

## Evaluation metrics and runtime
All 18 checkpoints are evaluated with one shared saved-prediction protocol:
square 640-pixel letterboxing, FP32, batch one, confidence floor 0.001, maximum
1,000 detections per image, and prediction IoU 0.50 where applicable.
Native end-to-end heads retain their architecture-specific postprocessing.
Class-aware matching and the pinned 101-point interpolated AP implementation
yield mAP50 and mAP50:95. Results are reported as mean ± sample standard
deviation across three seeds; SD is not a confidence interval or significance
test. Operating precision, recall and F1 pool class TP/FP/FN at confidence 0.25
and matching IoU 0.50. They are distinct from best-F1-threshold values.

Runtime is measured separately on an isolated NVIDIA H100 80GB with FP32
square-640, batch-one predictions. Each checkpoint receives 30 warmup and 100
measured predictions over cached decoded test images. Timings include
preprocessing, detector and postprocessing, and exclude disk decoding, model
loading and warmup. Parameter counts refer to stored unfused checkpoints.
No speedup against results measured on different hardware is claimed.

## Controlled original-label results
The crop-trained recipe obtains mAP50 of 55.08 ± 0.89% and
mAP50:95 of 39.62 ± 0.64%. Under the same evaluator,
YOLO26s obtains 49.94 ± 1.85% and 35.03 ± 1.12%, respectively.
The observed mean differences are 5.14 percentage points in mAP50 and
4.59 points in mAP50:95. Relative to the retrained YOLOv8s control
(52.34 ± 1.38% mAP50), the crop-trained recipe improves the observed mean
by 2.73 points. These are descriptive three-seed differences, not evidence
of statistical significance.

AGWD plus Squared-Dice obtains 53.23 ± 1.40% mAP50, while
YOLOv12s obtains 53.61 ± 2.39% and PhysOT obtains 52.27 ± 1.79%.
Thus the proposed recipes do not uniformly outperform every control; in
particular, PhysOT's mean mAP50 is slightly below the retrained YOLOv8s result.
Per-class results and fixed-threshold precision/recall are reported alongside
the aggregate values rather than selecting only favorable metrics.

## Exploratory native-resolution evidence
A separate retrospective audit evaluates four selected legacy checkpoints
and two fixed WBF combinations on 44 historical native-resolution exports.
The strongest audited single checkpoint, YOLO26s crop specialist, obtains
66.56% mAP50; four-model WBF obtains 68.74%,
an observed difference of 2.18 percentage
points. Its full prediction pipeline takes 115.97 ms per image
under the separate square-1280 FP32 H100 protocol. These checkpoints have
different training recipes and are not a controlled training ablation.

Subsequent annotation review confirmed class errors, missing boxes and shared
photographs across source splits. These native-resolution results are therefore
exploratory retrospective results, not final independent-holdout performance.
The separate three-seed no-extra-online-augmentation loss study also uses the
legacy data and rectangular batch-eight evaluation; it must not be merged with
the square batch-one champion audit.

## Comparison with published work
Zhou and Agaian (Information 2026, 17, 322) report 54.04% mAP50 for YOLOv8s
and 59.63% for Consensus Top-4. These author-reported values provide context.
Available author source uses inconsistent split paths, AP definitions,
confidence filtering and detection limits; exact image and annotation
identities remain unverified. In particular, our available validation
annotations contain 1,032 ICs, while the paper's class table lists 1,026.
Consequently, we do not claim a certified reproduction or direct numerical
improvement against those published scores. Our defensible improvement
comparison is against our retrained controls under the fixed common protocol.

## Data review and pending experiments
The first-pass r3 candidate applies 1,758 class changes, adds 95 boxes and
removes 25 empty-footprint boxes across 212 exports. Totals include repeated
transformed/resized views. It conservatively quarantines 56 training and two
validation exports from board families shared with higher-priority held-out
splits, leaving 492/78/44 active images. Original files remain unchanged.
All 672 source exports received overview inspection, but exhaustive
component-level verification, occluded-box policy and held-out duplicate
handling remain unfinished. Existing validation images also include upsampled
640-pixel exports, which do not provide restored native detail.

The five newly submitted r3 jobs and the six original-label AGWD/Exp7 jobs
have not supplied completed results for this notebook. No claim about accuracy
after correction, diffusion augmentation, a new Exp7 reproduction, or a new
ensemble is made. Final paper claims require a frozen dataset and evaluation
policy, completed experiments, and clear separation of legacy exploratory
evidence from the principal controlled comparison.

## Reference
X. Zhou and S. Agaian, 'Ensemble Learning Using YOLO Models for Semiconductor
E-Waste Recycling,' Information, 17, 322, 2026.
https://doi.org/10.3390/info17040322
