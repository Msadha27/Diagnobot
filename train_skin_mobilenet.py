from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "skin_diseases"
MODEL_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluation"
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 12
PATIENCE = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(targets: list[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[target].append(index)

    rng = random.Random(seed)
    train, validation, test = [], [], []
    for target in sorted(by_class):
        indices = by_class[target][:]
        rng.shuffle(indices)
        n = len(indices)
        n_train = round(0.70 * n)
        n_validation = round(0.15 * n)
        train.extend(indices[:n_train])
        validation.extend(indices[n_train:n_train + n_validation])
        test.extend(indices[n_train + n_validation:])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def confusion_metrics(y_true: list[int], y_pred: list[int], n_classes: int) -> tuple[list[list[int]], dict]:
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1

    per_class = []
    for i in range(n_classes):
        tp = matrix[i][i]
        support = sum(matrix[i])
        fp = sum(matrix[row][i] for row in range(n_classes)) - tp
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": support})

    accuracy = sum(matrix[i][i] for i in range(n_classes)) / max(1, len(y_true))
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": sum(x["recall"] for x in per_class) / n_classes,
        "macro_precision": sum(x["precision"] for x in per_class) / n_classes,
        "macro_recall": sum(x["recall"] for x in per_class) / n_classes,
        "macro_f1": sum(x["f1"] for x in per_class) / n_classes,
        "per_class": per_class,
    }
    return matrix, metrics


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int):
    model.eval()
    y_true, y_pred = [], []
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += criterion(logits, labels).item() * labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    matrix, metrics = confusion_metrics(y_true, y_pred, n_classes)
    metrics["loss"] = loss_total / max(1, len(y_true))
    return matrix, metrics, y_true, y_pred


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_total = 0.0
    correct = 0
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return loss_total / max(1, seen), correct / max(1, seen)


def write_matrix(path: Path, matrix: list[list[int]], class_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ predicted", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row])


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()

    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Separate ImageFolder instances prevent train augmentation from leaking into validation/test transforms.
    base = datasets.ImageFolder(str(DATA_DIR))
    class_names = base.classes
    train_indices, val_indices, test_indices = stratified_split(base.targets, SEED)
    train_base = datasets.ImageFolder(str(DATA_DIR), transform=train_tfms)
    eval_base = datasets.ImageFolder(str(DATA_DIR), transform=eval_tfms)
    train_ds = Subset(train_base, train_indices)
    val_ds = Subset(eval_base, val_indices)
    test_ds = Subset(eval_base, test_indices)

    workers = 0  # Reliable on Windows; images are small and the dataset is modest.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers)

    print(f"Device: {device}", flush=True)
    print(f"Classes ({len(class_names)}): {class_names}", flush=True)
    print(f"Split sizes: train={len(train_ds)}, validation={len(val_ds)}, test={len(test_ds)}", flush=True)
    print("Loading ImageNet-pretrained MobileNetV3-Small weights...", flush=True)
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
    model.to(device)

    counts = np.bincount([base.targets[i] for i in train_indices], minlength=len(class_names))
    class_weights = torch.tensor(np.sqrt(counts.sum() / counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=5e-4, weight_decay=1e-4)

    candidate_path = MODEL_DIR / "skin_mobilenet_candidate.pt"
    history_path = EVAL_DIR / "skin_training_history.csv"
    history = []
    best_f1 = -1.0
    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    no_improve = 0
    epoch_number = 0

    def run_stage(epochs: int, stage: str, optimizer):
        nonlocal best_f1, best_val_loss, best_state, best_epoch, no_improve, epoch_number
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        for _ in range(epochs):
            epoch_number += 1
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            _, val_metrics, _, _ = evaluate(model, val_loader, device, len(class_names))
            scheduler.step(val_metrics["macro_f1"])
            row = {
                "epoch": epoch_number,
                "stage": stage,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"Epoch {epoch_number:02d} [{stage}] train_acc={train_acc:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"val_loss={val_metrics['loss']:.4f}",
                flush=True,
            )
            improved = (val_metrics["macro_f1"] > best_f1 + 1e-6) or (
                abs(val_metrics["macro_f1"] - best_f1) <= 1e-6 and val_metrics["loss"] < best_val_loss
            )
            if improved:
                best_f1 = val_metrics["macro_f1"]
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch_number
                no_improve = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                # Persist each validation-best state so an interrupted run still leaves a usable candidate.
                torch.save({
                    "model_state_dict": best_state,
                    "class_names": class_names,
                    "img_size": IMAGE_SIZE,
                    "architecture": "mobilenet_v3_small",
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_f1,
                    "best_val_loss": best_val_loss,
                    "seed": SEED,
                }, candidate_path)
            else:
                no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping after {PATIENCE} validation epochs without improvement.", flush=True)
                break

    run_stage(WARMUP_EPOCHS, "head", optimizer)

    # Fine-tune only the final MobileNet feature blocks to limit overfitting on this small dataset.
    feature_blocks = list(model.features.children())
    for param in model.features.parameters():
        param.requires_grad = False
    for block in feature_blocks[-4:]:
        for param in block.parameters():
            param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=1e-4
    )
    run_stage(FINETUNE_EPOCHS, "fine_tune", optimizer)

    if best_state is None:
        raise RuntimeError("Training did not produce a best validation checkpoint")
    model.load_state_dict(best_state)
    test_matrix, test_metrics, y_true, y_pred = evaluate(model, test_loader, device, len(class_names))

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "img_size": IMAGE_SIZE,
        "architecture": "mobilenet_v3_small",
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_val_loss": best_val_loss,
        "seed": SEED,
    }, candidate_path)

    with history_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    write_matrix(EVAL_DIR / "skin_test_confusion_matrix.csv", test_matrix, class_names)
    report = {
        "dataset": str(DATA_DIR.relative_to(ROOT)),
        "classes": class_names,
        "split_method": "stratified image-level 70/15/15; no patient identifiers were available",
        "split_seed": SEED,
        "split_counts": {
            "train": len(train_indices),
            "validation": len(val_indices),
            "test": len(test_indices),
        },
        "class_counts": {
            name: {
                "train": int(sum(base.targets[i] == class_id for i in train_indices)),
                "validation": int(sum(base.targets[i] == class_id for i in val_indices)),
                "test": int(sum(base.targets[i] == class_id for i in test_indices)),
            }
            for class_id, name in enumerate(class_names)
        },
        "selection": {"best_epoch": best_epoch, "criterion": "validation macro F1; test split not used for selection"},
        "test_metrics": test_metrics,
        "per_class_test_metrics": {
            name: test_metrics["per_class"][i] for i, name in enumerate(class_names)
        },
        "training_seconds": round(time.time() - started, 2),
        "checkpoint": str(candidate_path.relative_to(ROOT)),
        "limitations": [
            "Small single-source dataset; metrics have substantial uncertainty.",
            "Image-level split only; no patient/source metadata was available to prevent related-image leakage.",
            "This evaluates image class prediction only, not triage decisions or generated clinical answers.",
            "Research prototype only; not validated for clinical use.",
        ],
    }
    with (EVAL_DIR / "skin_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nUntouched test metrics:", flush=True)
    print(json.dumps({k: v for k, v in test_metrics.items() if k != "per_class"}, indent=2), flush=True)
    print(f"Best validation epoch: {best_epoch}; checkpoint saved to {candidate_path}", flush=True)
    print(f"Evaluation artifacts: {EVAL_DIR}", flush=True)


if __name__ == "__main__":
    main()
