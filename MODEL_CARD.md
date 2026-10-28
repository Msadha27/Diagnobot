# Skin classifier model card

## Summary

- **Task:** classify a skin image into one of the ten dataset folder labels.
- **Architecture:** torchvision MobileNetV3-Small, initialized with ImageNet default weights and fine-tuned on this dataset.
- **Input:** RGB image resized to 224 × 224 and ImageNet normalized.
- **Training script:** `train_skin_mobilenet.py`.
- **Active checkpoint:** `models/skin_mobilenet.pt`.
- **Previous checkpoint:** `models/skin_mobilenet_previous.pt`.
- **Selection criterion:** best validation macro-F1, followed by one evaluation on the held-out test split.
- **Intended use:** student prototype and engineering demonstration only; not diagnosis or treatment.

## Dataset and split

The local dataset has 932 images in ten class folders. A deterministic, class-stratified split with seed 42 was made at the image level:

| Partition | Images |
|---|---:|
| Training | 652 |
| Validation | 140 |
| Internal test | 140 |

There are 63 `acanthosis nigricans`, 72 `impetigo`, 97 `molluscum-contagiosum`, and 100 images in each other class. Patient identity and acquisition-source metadata were unavailable, so the split cannot rule out related images in multiple partitions. Dataset origin, licensing, image provenance, labeler qualifications, and reference-label procedure are not sufficiently documented in this project. Resolve these before making paper claims.

## Training method

The training code uses separate dataset instances so training augmentation does not affect validation or test. Training augmentation includes horizontal flips, small rotations, and mild color changes. It trains the classifier head first, then fine-tunes only the final MobileNet feature blocks at a lower learning rate. It uses class-weighted cross-entropy, weight decay, a learning-rate scheduler, and early stopping on validation macro-F1. The best checkpoint is saved whenever validation performance improves.

## Results

The untouched internal image-level test split produced:

| Metric | Result |
|---|---:|
| Accuracy | 0.7071 |
| Balanced accuracy | 0.7031 |
| Macro precision | 0.7181 |
| Macro recall | 0.7031 |
| Macro-F1 | 0.6993 |
| Cross-entropy loss | 0.9154 |

Best validation macro-F1 was 0.6760 at epoch 15. Training accuracy at that epoch was 0.8175 and validation accuracy was 0.6714. The remaining difference suggests some overfitting may persist. A rough 95% Wilson interval for 99 correct predictions among 140 test images is approximately 62.8%–77.7%; this does not account for possible image leakage or dataset bias. Treat the result as preliminary engineering evidence, not clinical diagnostic performance.

Per-class metrics are in `evaluation/skin_test_metrics.json`; errors are summarized by `evaluation/skin_test_confusion_matrix.csv`.

## Label set

These are the dataset folder names: Bullous; Eczema Photos; Lupus; Urticaria Hives; acanthosis nigricans; acne; candidiasis; impetigo; molluscum-contagiosum; nevus. The labels may not represent mutually exclusive clinical diagnoses. Confirm definitions and the reference standard before interpreting class performance.

## Limitations

- Small dataset and only 140 internal-test images.
- Image-level, not patient-level split; duplicate/related-image leakage is possible.
- No external institution or device test set.
- Image provenance, consent, labeler expertise, and reference standard are not fully documented.
- Each class has only about 10–15 test images.
- No completed calibration analysis or clinically chosen operating threshold.
- No evaluation of LLM responses, triage rules, demographic subgroups, workflow, or patient outcomes.
- The application has a fixed low-confidence abstention behavior, but it has not been calibrated against a representative clinical cohort.

## Improvement plan

1. **Define one primary research question.** Choose a coherent task and classes with consistent definitions and reliable labels. Narrowing scope can make the evaluation meaningful, but does not automatically raise accuracy.
2. **Audit data first.** Verify labels with qualified reviewers, remove corrupt files, detect exact and near duplicates, document source/device and conditions, and retain patient/source grouping identifiers where governance allows.
3. **Increase independent data.** Collect representative images across patients, skin tones, devices, lighting, and sites. Keep every patient/source entirely in one split.
4. **Freeze the final test cohort.** Tune only on training/validation data; use a separate preferably external cohort for final evaluation.
5. **Add baselines.** Compare against majority-class and at least one suitable pretrained model with identical splits/preprocessing. If external testing is not yet possible, use grouped cross-validation and report uncertainty.
6. **Tune carefully.** Select learning rate, weight decay, augmentation, resolution, and class weighting on grouped validation only. Keep clinically important appearance intact; apply focal loss or label smoothing only when error analysis justifies it.
7. **Report useful metrics.** Include confusion matrix, class-wise sensitivity/recall, specificity, precision, macro-F1, balanced accuracy, suitable one-vs-rest ROC-AUC, calibration (e.g. Brier score), confidence intervals, and abstention coverage/error. State the reference standard and thresholds.
8. **Evaluate modules separately.** Have clinicians assess LLM summaries and triage with a prespecified rubric. Skin-classifier accuracy is not LLM-answer accuracy or whole-system accuracy.
9. **Follow reporting guidance.** Use CLAIM 2024 and transparently report cohort, provenance, labels, splitting unit, exclusions, preprocessing, selection, uncertainty, internal and external testing. This image-level holdout is not external testing.

## Reproduction

From the project root, activate the Python environment and run:

```powershell
python train_skin_mobilenet.py
```

The script writes `models/skin_mobilenet_candidate.pt`, `evaluation/skin_training_history.csv`, `evaluation/skin_test_confusion_matrix.csv`, and `evaluation/skin_test_metrics.json`. Record the dataset version, package versions, training hardware, pretrained-weight identifier, and code revision for each run.
