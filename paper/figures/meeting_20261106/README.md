# Figure availability and scope

All 16 figures were generated from saved evidence, not reference-paper screenshots.
Primary test comparisons use three seeds. PR curves, confusion matrices and
qualitative examples explicitly use seed 0.
No diffusion augmentation, DCD-Net comparison, new Exp7 result or corrected-r3
accuracy is fabricated. Exact published-paper equivalence remains unverified.
Legacy native-resolution numbers carry the known split/annotation limitations.

## 01_study_design: Study design and model recipes
The completed original-label experiment compares six methods across three seeds. The crop method changes training sampling; all methods use a single full-board pass during common evaluation. Module indicators describe recipe differences and are not an isolated module ablation.

Scope: Completed original-label study. R3-corrected training and original_top2_20261105 Exp7 jobs are excluded.

## 02_dataset_examples_counts: Dataset examples and annotation counts
Four canonical component categories and original-export instance counts. Illustrative crops are the first test annotations, in recorded order, with a short side of at least eight pixels; they were not selected using detector performance. Existing original labels are preserved.

Scope: Original-label RF100 version 3 exports. Total: 543 train / 77 validation / 41 test images. Inherited split; no board-disjoint generalization claim.

## 03_primary_accuracy: Shared-protocol test accuracy
Mean and sample SD across seeds 0/1/2 on the 41-image original-label test split. Hollow markers show individual seeds. Each model is evaluated with the same common scorer and full-board input, including the crop-trained method. No significance claim is made from three seeds.

Scope: Original-label test · square 640 · FP32 · conf floor 0.001 · maxDet 1000 · common AP101. Inherited split; no board-disjoint generalization claim.

## 04_class_and_operating_metrics: Per-class AP and operating metrics
AP is averaged over three seeds. Pooled precision and recall are computed by summing class TP/FP/FN at confidence 0.25 and IoU 0.50 within each seed, then summarizing across seeds. They are not best-F1-threshold or macro-averaged P/R. The test split contains only 18 electrolytic annotations.

Scope: Original-label common test protocol · three seeds · AP50 uses all predictions above the 0.001 input floor. Inherited split; no board-disjoint generalization claim.

## 05_precision_recall: Class-specific precision–recall curves
Seed-0 test PR precision envelopes from frozen common-profile predictions, using the pinned scorer's class-aware IoU matching and AP101 endpoint convention. Parenthesized values are this seed's class AP50, independently reproduced against saved metrics. These curves are not three-seed means.

Scope: Original-label test · seed 0 diagnostics · full-board prediction · same thresholds and caps for all methods. Inherited split; no board-disjoint generalization claim.

## 06_confusion_matrices: Diagnostic confusion matrices
Greedy one-to-one class-agnostic IoU matching at IoU 0.50 and confidence 0.25. Rows denote predicted classes; columns denote annotation classes. Background row counts missed annotations; background column counts unmatched detections. Each column is normalized separately, so the background column is an FP composition, not recall. Raw counts are exported. This matrix is not the class-aware operating-metric matcher.

Scope: Original-label common predictions · seed 0 · true class denominators include only 18 electrolytic boxes. Inherited split; no board-disjoint generalization claim.

## 07_detection_comparison: Full-board detection comparisons
Two predeclared filename groups (Arty_Top and pcb125rec1), using the first export in frozen record order. Columns show original annotations and seed-0 predictions; only predictions with confidence at least 0.25 are displayed. Colors encode class, not correctness. Same image, crop and display threshold are used in every column. Unannotated detections are not automatically false positives because source labels may be incomplete.

Scope: Original-label 640 test illustrations · cached predictions · images selected independently of detector scores. Inherited split; no board-disjoint generalization claim.

## 08_small_component_zoom: Small-component comparison with fixed zoom
The same Arty_Top export as Figure 7, with ROI (0, 0, 256, 256). The region maximizes annotated capacitor centers over a fixed 256-pixel window grid; detector performance is not used for selection. Red outlines indicate the shared ROI. Enlarged display does not add image resolution. Prediction confidence is at least 0.25; boxes are cached seed-0 outputs rather than manually redrawn detections.

Scope: Original-label seed-0 diagnostic · ROI selected from annotation density · fixed region across methods. Inherited split; no board-disjoint generalization claim.

## 09_training_dynamics: Completed training dynamics
Mean validation mAP50 and box-loss traces from the actual completed CSV logs, seeds 0/1/2. Full-board methods use 100 epochs and mixed-crop training 50; the x-axis is normalized scheduled budget, not an equal epoch count. Training-time rectangular/batched validation differs from the common square batch-one final evaluator. Different box-loss definitions mean their absolute loss magnitudes are not directly comparable.

Scope: Completed original-label training histories · three seeds · no smoothing or fabricated r3 curves.

## 10_accuracy_speed_parameters: Accuracy, latency and model size
The same 18 checkpoints used in the primary accuracy comparison were independently benchmarked on an isolated NVIDIA H100 80GB, FP32, square 640 and batch one. Thirty warmups and 100 timed predictions per checkpoint; latency includes preprocessing, detector and postprocessing, excluding disk decoding/model loading. Error bars are SD across the three checkpoint means. Counts are stored unfused parameters, not FLOPs.

Scope: Shared original-label accuracy · isolated H100 FP32 timing · no speed comparison with the paper's Apple M4. Inherited split; no board-disjoint generalization claim.

## 11_legacy_ensembles: Exploratory native-resolution ensemble audit
Four fixed checkpoints and two fixed WBF combinations on the original 44-export legacy test split, square 1280 and FP32. WBF weights are 1.3/1.2 and 1.3/1.2/1.1/1.0, merge IoU 0.50. No new ensemble tuning occurs. These are selected checkpoints with different training recipes. Later review confirmed cross-split photograph overlap and label defects; the scores are exploratory and not certified generalization results.

Scope: EXPLORATORY LEGACY DATA · 44 exports, not 44 independent boards · r3 corrections were not applied to these scores.

## 12_legacy_loss_comparison: Historical three-seed loss comparison
Related YOLO26s loss recipes from the completed no-extra-online-augmentation native study, mean ± sample SD over seeds 0/1/2. Squared-Dice smoothing is 0.01. This study used rectangular batch-8 FP32 evaluation, not the selected-champion square batch-one audit. All 21 results, including seven methods and negative comparisons, remain in the source summary; this panel selects the four related YOLO26s loss recipes by architecture rather than score. Shared legacy split/annotation limitations remain.

Scope: EXPLORATORY LEGACY ABLATION · original native labels · rect=True, batch 8 · separate from Figures 3–10.

## 13_annotation_corrections: Traced annotation correction examples
Top: an existing class-0 annotation changed to class 2, retaining its coordinates. Bottom: four clearly identifiable capacitors that lacked source boxes. These are traced label edits, not model predictions. The full review records 1,758 class changes, 95 additions and 25 deletions across 212 exports. Those totals include transformed/resized variants, not independent objects. R3 component-level review is incomplete; no new training accuracy is inferred from this illustration.

Scope: DATA-QUALITY REVIEW ONLY · original files preserved · no r3 accuracy in this snapshot · no model improvement claimed.

## 14_published_context: Published context with explicit comparability limits
Published values are author-reported from Zhou and Agaian, Information 2026, 17, 322, doi:10.3390/info17040322, Tables 4 and 9. Our results use the available original-label RF100 exports and a frozen shared evaluator. Exact author image/annotation identities and table-specific evaluation settings remain unresolved. No cross-block improvement, ranking or speedup claim is justified.

Scope: CONTEXT ONLY across the two blocks. Quantitative improvement claims use our own identically evaluated controls.

## 15_native_master_metrics: Exact native champion and ensemble metrics
All six completed legacy configurations under the same square-1280 batch-one FP32 audit. P/R/F1 pool counts at confidence 0.25 and matching IoU 0.50; AP uses all saved predictions. Latency includes preprocessing, detector, postprocessing and WBF fusion; FPS is 1,000 divided by mean latency. Parameters sum the stored unfused member checkpoints. Class AP values include the legacy electrolytic category with only 18 annotations. Known cross-split photographs and labeling defects preclude a final independent-generalization claim.

Scope: EXPLORATORY LEGACY AUDIT · exact checkpoint hashes · no mixing with original-label or corrected-r3 metrics.

## 16_native_ensemble_detections: Legacy single-model and WBF detections
One predeclared Arty_Top legacy export and its annotation-density-selected zoom. Predictions are exactly the cached audit outputs, with confidence at least 0.25 for display. WBF confidence rescaling is preserved; equal numeric confidence does not imply identical score calibration across pipelines. Original image/label hashes and all three caches were checked, and their class AP50 and operating TP/FP counts independently reproduced. Colors denote class, not correctness. Native split overlap and annotation limitations remain.

Scope: EXPLORATORY LEGACY DATA · selected checkpoint audit · full board and same zoom · no new inference or retuning.