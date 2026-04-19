# Binary classifier for BreaKHis 256x256 histopathology crops (benign vs malignant).
#
# Based on PyTorch's official "Transfer Learning for Computer Vision Tutorial"
# by Sasank Chilamkurthi:
#   https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#
# The tutorial fine-tunes a pretrained ResNet-18 on a small binary classification
# dataset (ants vs bees, ~240 images). We follow the same approach for BreaKHis
# breast cancer histopathology crops.
#
# How to run (from the project root, with venv activated):
#   python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir <path> --outdir <dir>
#
# Examples:
#   # Train on real images only:
#   python project/pipeline/03-classifier-experiments/train_classifier.py \
#       --data-dir project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops/40X \
#       --outdir project/classifier-runs/40x/real-only-default
#
#   # Train on real + balanced synthetic images:
#   python project/pipeline/03-classifier-experiments/train_classifier.py \
#       --data-dir project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops/40X \
#       --synthetic-dir project/synthetic-images/40x-snapshot-008800 \
#       --outdir project/classifier-runs/40x/real-and-synthetic
#
#   # Train with class weighting to handle imbalance:
#   python project/pipeline/03-classifier-experiments/train_classifier.py \
#       --data-dir project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops/40X \
#       --class-weights \
#       --outdir project/classifier-runs/40x/real-only-class-weights
#
# Expected data structure:
#   --data-dir: {train,validation,test}/{benign,malignant}/*.png
#   --synthetic-dir (optional): {benign,malignant}/*.png

import argparse
import os
import re
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix


class TeeOutput:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------
# Training: resize to 224x224 (ResNet-18's expected input size) with random
# flips for augmentation. Horizontal and vertical flips are standard for
# histopathology since tissue under a microscope has no orientation.
#
# Validation/Test: resize only, no augmentation. Evaluation must be
# deterministic — the same image should always produce the same prediction.
#
# Both use ImageNet normalisation (mean and std computed over the 1.2M ImageNet
# training images). This is required because ResNet-18 was pretrained with
# these values — the pretrained weights expect inputs normalised this way.
# ---------------------------------------------------------------------------

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(224),              # resize 256x256 -> 224x224
        transforms.RandomHorizontalFlip(),   # 50% chance horizontal flip
        transforms.RandomVerticalFlip(),     # 50% chance vertical flip
        transforms.ToTensor(),               # convert PIL image to tensor [0, 1]
        transforms.Normalize(                # normalise with ImageNet statistics
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(224),              # resize 256x256 -> 224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
# Follows the tutorial's train_model() structure:
#   - Each epoch has a training phase and a validation phase
#   - model.train() sets the model to training mode
#   - model.eval() sets the model to evaluation mode
#   - torch.set_grad_enabled(phase == 'train') avoids computing gradients
#     during validation, saving memory and computation
#   - The best model (by validation accuracy) is saved to disk
#   - scheduler.step() is called once per epoch after the training phase
# ---------------------------------------------------------------------------

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer,
                scheduler, device, num_epochs, outdir):
    since = time.time()

    best_acc = 0.0
    best_model_path = os.path.join(outdir, "best_model.pth")
    history = []

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Each epoch has a training and validation phase
        # (following the tutorial's structure)
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()   # set model to training mode
            else:
                model.eval()    # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Only track gradients during training (tutorial pattern)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimise only during training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Accumulate statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Step the learning rate scheduler after training phase
            # (following the tutorial: scheduler.step() once per epoch)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss, train_acc = epoch_loss, epoch_acc.item()
            else:
                val_loss, val_acc = epoch_loss, epoch_acc.item()

        # Print epoch summary
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:>3d}/{num_epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={current_lr:.6f}  time={elapsed:.1f}s")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        # Save the best model (by validation accuracy), following the tutorial
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best val_acc: {val_acc:.4f}")

    total_time = time.time() - since
    print(f"\nTraining complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best val acc: {best_acc:.4f}")

    # Load best model weights (following the tutorial)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model, history


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------
# The tutorial only evaluates on a validation set. We add a separate test set
# evaluation with a full classification report (precision, recall, F1 per
# class) and confusion matrix, which is standard practice for reporting
# results in machine learning.
# ---------------------------------------------------------------------------

def evaluate_test(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    report_dict = classification_report(all_labels, all_preds,
                                        target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, report, report_dict, cm


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------
# When the dataset is imbalanced (e.g. 30% benign, 70% malignant), the model
# can achieve high accuracy by always predicting the majority class. Class
# weighting addresses this by making the loss function penalise errors on the
# minority class more heavily.
#
# Weight for class i = total_samples / (num_classes * samples_in_class_i)
#
# This is the inverse-frequency weighting scheme, which is a standard
# technique for handling class imbalance. PyTorch's CrossEntropyLoss accepts
# a weight tensor via the `weight` parameter.
# See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# ---------------------------------------------------------------------------

def compute_class_weights(dataset):
    """Compute inverse-frequency class weights from dataset targets."""
    targets = dataset.targets
    class_counts = np.bincount(targets)
    total = len(targets)
    num_classes = len(class_counts)
    weights = total / (num_classes * class_counts)
    print(f"Class counts: {dict(enumerate(class_counts))}")
    print(f"Class weights: {dict(enumerate(weights.round(4)))}")
    return torch.FloatTensor(weights)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ResNet-18 binary classifier for BreaKHis histopathology crops"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Path to real data (with train/validation/test subfolders)")
    parser.add_argument("--synthetic-dir", default=None,
                        help="Path to synthetic image directory (with class subfolders, e.g. benign/)")
    parser.add_argument("--synth-malignant", type=int, default=0,
                        help="Number of synthetic malignant images to add. "
                             "Synthetic benign count is computed automatically to "
                             "balance classes: real_benign + synth_benign = "
                             "real_malignant + synth_malignant. Default: 0.")
    parser.add_argument("--outdir", required=True,
                        help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs (default: 25, same as tutorial)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate (default: 0.001, same as tutorial)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay (L2 regularisation) for SGD (default: 0.0)")
    parser.add_argument("--class-weights", action="store_true",
                        help="Use inverse-frequency class weighting in loss")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze all layers except the final FC layer")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout probability before final FC layer (default: 0.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sys.stdout = TeeOutput(os.path.join(args.outdir, "log.txt"))

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------
    # Setting seeds ensures the same random initialisations, data shuffling,
    # and augmentations across runs. cudnn.deterministic ensures deterministic
    # behaviour on GPU at the cost of some performance.
    # -----------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------------
    # Using torchvision.datasets.ImageFolder, which expects:
    #   root/{class_name}/*.png
    # Classes are assigned alphabetically: benign=0, malignant=1
    # This is the same approach used in the PyTorch tutorial.
    # -----------------------------------------------------------------------
    train_transform, eval_transform = get_transforms()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "validation")
    test_dir = os.path.join(args.data_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = train_dataset.classes
    real_benign = sum(1 for _, c in train_dataset.samples if c == train_dataset.class_to_idx["benign"])
    real_malignant = sum(1 for _, c in train_dataset.samples if c == train_dataset.class_to_idx["malignant"])
    print(f"Classes: {class_names}")
    print(f"Real training: {real_benign} benign, {real_malignant} malignant")

    # Track image counts for results
    synth_benign = 0
    synth_malignant = 0

    # Optionally add balanced synthetic data to the training set.
    # A temporary symlink directory is created inside --outdir containing only
    # the selected synthetic images, so ImageFolder loads only what is needed.
    if args.synthetic_dir is not None:
        # Sort filenames numerically (e.g. seed9.png before seed10.png)
        def numerical_sort(filenames):
            def sort_key(f):
                nums = re.findall(r'\d+', f)
                return int(nums[0]) if nums else 0
            return sorted(filenames, key=sort_key)

        synth_benign_available = numerical_sort(os.listdir(os.path.join(args.synthetic_dir, "benign")))
        synth_malignant_available = numerical_sort(os.listdir(os.path.join(args.synthetic_dir, "malignant"))) \
            if os.path.isdir(os.path.join(args.synthetic_dir, "malignant")) else []

        # Clamp synth_malignant to available images first
        synth_malignant = min(args.synth_malignant, len(synth_malignant_available))
        if args.synth_malignant > len(synth_malignant_available):
            print(f"WARNING: Not enough synthetic malignant images. "
                  f"Requested {args.synth_malignant}, have {len(synth_malignant_available)}. "
                  f"Using {synth_malignant}.")
        # Compute synthetic benign for class balance:
        # real_benign + synth_benign = real_malignant + synth_malignant
        synth_benign_needed = (real_malignant + synth_malignant) - real_benign
        synth_benign = max(0, min(synth_benign_needed, len(synth_benign_available)))

        if synth_benign_needed > len(synth_benign_available):
            print(f"WARNING: Not enough synthetic benign images to balance classes. "
                  f"Need {synth_benign_needed}, have {len(synth_benign_available)}. "
                  f"Classes will be imbalanced by {synth_benign_needed - len(synth_benign_available)} images.")

        print(f"Synthetic available: {len(synth_benign_available)} benign, {len(synth_malignant_available)} malignant")
        print(f"Adding {synth_benign} synth benign + {synth_malignant} synth malignant")

        # Create temporary symlink directory with balanced synthetic images
        synth_link_dir = os.path.join(args.outdir, "_synthetic_balanced")
        synth_ben_dir = os.path.join(synth_link_dir, "benign")
        os.makedirs(synth_ben_dir, exist_ok=True)
        for fname in synth_benign_available[:synth_benign]:
            src = os.path.abspath(os.path.join(args.synthetic_dir, "benign", fname))
            dst = os.path.join(synth_ben_dir, fname)
            if not os.path.exists(dst):
                os.symlink(src, dst)

        if synth_malignant > 0:
            synth_mal_dir = os.path.join(synth_link_dir, "malignant")
            os.makedirs(synth_mal_dir, exist_ok=True)
            for fname in synth_malignant_available[:synth_malignant]:
                src = os.path.abspath(os.path.join(args.synthetic_dir, "malignant", fname))
                dst = os.path.join(synth_mal_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(src, dst)

        synthetic_dataset = datasets.ImageFolder(synth_link_dir, transform=train_transform)
        print(f"Synthetic images loaded: {len(synthetic_dataset)}")
        train_dataset = ConcatDataset([train_dataset, synthetic_dataset])
        total_benign = real_benign + synth_benign
        total_malignant = real_malignant + synth_malignant
        print(f"Total training: {total_benign} benign + {total_malignant} malignant = {len(train_dataset)}")

    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    # Dataset sizes used for computing epoch-level averages
    dataset_sizes = {
        'train': len(train_dataset),
        'validation': len(val_dataset),
    }

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True),
        'validation': DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True),
    }
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # -----------------------------------------------------------------------
    # Model setup
    # -----------------------------------------------------------------------
    # Following the tutorial: load a pretrained ResNet-18 and replace the
    # final fully connected layer for binary classification.
    #
    # ResNet-18 has ~11.7M parameters pretrained on ImageNet (1.2M images,
    # 1000 classes). The final FC layer (512 -> 1000) is replaced with
    # (512 -> 2) for our binary task. The new layer is randomly initialised.
    # -----------------------------------------------------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Optionally freeze the backbone (all layers except the final FC layer).
    # This means only the new classifier head is trained, using the pretrained
    # ImageNet features as a fixed feature extractor.
    # See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    #      ("ConvNet as fixed feature extractor" section)
    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    if args.dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(num_ftrs, len(class_names)),
        )
    else:
        model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # -----------------------------------------------------------------------
    # Loss function
    # -----------------------------------------------------------------------
    # CrossEntropyLoss is the standard loss for multi-class (including binary)
    # classification, as used in the tutorial.
    #
    # When --class-weights is set, we apply inverse-frequency weighting to
    # handle class imbalance. This is done via the `weight` parameter of
    # CrossEntropyLoss (see PyTorch docs).
    # -----------------------------------------------------------------------
    if args.class_weights:
        # Compute weights from the real training set (before synthetic addition)
        real_train = datasets.ImageFolder(train_dir)
        weights = compute_class_weights(real_train).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------------------------
    # Optimiser and scheduler
    # -----------------------------------------------------------------------
    # Following the tutorial:
    #   - SGD with momentum 0.9 and lr=0.001
    #   - StepLR: multiply lr by 0.1 every 7 epochs
    #
    # SGD with momentum is the standard optimiser for fine-tuning pretrained
    # CNNs. The learning rate is decayed by a factor of 10 every 7 epochs
    # to allow the model to converge to a finer minimum as training progresses.
    #
    # Weight decay (L2 regularisation) penalises large weights to reduce
    # overfitting. The PyTorch ImageNet example uses weight_decay=1e-4.
    # See: https://github.com/pytorch/examples/tree/main/imagenet
    # -----------------------------------------------------------------------
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    step_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs...")
    model, history = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer,
        step_scheduler, device, args.epochs, args.outdir
    )

    # -----------------------------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------------------------
    print("\nEvaluating best model on test set...")
    test_acc, report, report_dict, cm = evaluate_test(
        model, test_loader, device, class_names
    )

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"\nClassification report:\n{report}")
    print(f"Confusion matrix:\n{cm}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "test_accuracy": test_acc,
        "best_val_accuracy": max(h["val_acc"] for h in history),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "image_counts": {
            "real_benign": real_benign,
            "real_malignant": real_malignant,
            "synth_benign": synth_benign,
            "synth_malignant": synth_malignant,
            "total_train": real_benign + real_malignant + synth_benign + synth_malignant,
            "validation": len(val_dataset),
            "test": len(test_dataset),
        },
        "args": vars(args),
        "history": history,
    }

    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.outdir}/")


if __name__ == "__main__":
    main()
