"""
40/20/40 synthetic augmentation experiment.

Uses a 40/20/40 patient-level stratified split (40% train, 20% val, 40% test)
to create a large, reliable test set with limited training data. Compares
real-only vs real+synthetic classifier performance.
Synthetic balancing is handled by train_classifier.py via --synth-malignant.

Steps:
  1. Gather all crops from --data-dir and map to patient IDs
  2. Stratified 40/20/40 patient split (train/val/test)
  3. Create symlinked directory structures for real data
  4. Run train_classifier.py: real-only and real+synthetic
  5. Compare results

Usage (from project root):
    python project/pipeline/03-classifier-experiments/02-split-40-20-40/40_20_40_synthetic_augmentation_experiment.py \
        --outdir project/classifier-runs/40x/40-20-40-synthetic-augmentation-experiment \
        --data-dir project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops/40X \
        --synthetic-dir project/synthetic-images/40x-snapshot-008800
"""

import argparse
import os
import sys
import json
import numpy as np
from collections import defaultdict
import subprocess
import time


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


# -------------------------------------------------------------------------
# CLI arguments
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Synthetic augmentation experiment with 40/20/40 split"
)
parser.add_argument("--outdir", required=True,
                    help="Output directory for experiment results")
parser.add_argument("--data-dir", required=True,
                    help="Path to the 256x256 crop images organised by split "
                         "and class. Expected structure: "
                         "{data-dir}/{train,validation,test}/{benign,malignant}/*.png. "
                         "All crops from all splits are combined and "
                         "re-split into a 40/20/40 patient-level split.")
parser.add_argument("--synthetic-dir", required=True,
                    help="Path to GAN-generated synthetic images with class "
                         "subfolders (e.g. {synthetic-dir}/{benign,malignant}/*.png).")
parser.add_argument("--synth-malignant", type=int, default=0,
                    help="Number of synthetic malignant images to add. "
                         "Synthetic benign count is computed automatically to "
                         "balance classes. Default: 0.")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for patient shuffling (default: 42)")
args = parser.parse_args()

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
CROP_BASE = args.data_dir
SYNTHETIC_DIR = args.synthetic_dir
EXPERIMENT_BASE = args.outdir
SEED = args.seed

TRAIN_RATIO = 0.40
VAL_RATIO = 0.20
TEST_RATIO = 0.40

os.makedirs(EXPERIMENT_BASE, exist_ok=True)
sys.stdout = TeeOutput(os.path.join(EXPERIMENT_BASE, "log.txt"))


def get_patient_id(filename):
    """Extract patient ID using first three dash-separated tokens.
    e.g. SOB_B_A-14-22549CD-40-001-crop1.png -> SOB_B_A-14-22549CD
    Matches the original patient_split.py logic."""
    parts = filename.split("-")
    return "-".join(parts[:3])


# -------------------------------------------------------------------------
# Step 1: Gather all crops and map to patients
# -------------------------------------------------------------------------
print("Step 1: Gathering all crops and mapping to patients...")

patient_crops = defaultdict(list)   # patient_id -> [(abs_crop_path, class)]
patient_class = {}                  # patient_id -> "benign" or "malignant"

for split in ["train", "validation", "test"]:
    for cls in ["benign", "malignant"]:
        crop_dir = os.path.join(CROP_BASE, split, cls)
        if not os.path.isdir(crop_dir):
            continue
        for fname in sorted(os.listdir(crop_dir)):
            if not fname.endswith(".png"):
                continue
            pid = get_patient_id(fname)
            crop_path = os.path.abspath(os.path.join(crop_dir, fname))
            patient_crops[pid].append((crop_path, cls))
            patient_class[pid] = cls

benign_patients = sorted([p for p, c in patient_class.items() if c == "benign"])
malignant_patients = sorted([p for p, c in patient_class.items() if c == "malignant"])

print(f"  Total patients: {len(patient_class)}")
print(f"  Benign patients: {len(benign_patients)}")
print(f"  Malignant patients: {len(malignant_patients)}")
print(f"  Total crops: {sum(len(v) for v in patient_crops.values())}")

# -------------------------------------------------------------------------
# Step 2: Stratified 40/20/40 patient split
# -------------------------------------------------------------------------
print(f"\nStep 2: Creating {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)} stratified split (seed={SEED})...")

np.random.seed(SEED)

benign_shuffled = benign_patients.copy()
malignant_shuffled = malignant_patients.copy()
np.random.shuffle(benign_shuffled)
np.random.shuffle(malignant_shuffled)


def split_patients(patients, train_ratio, val_ratio):
    """Split patients into train/val/test using floor for train and val."""
    n = len(patients)
    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    n_test = n - n_train - n_val
    return (patients[:n_train],
            patients[n_train:n_train + n_val],
            patients[n_train + n_val:])


b_train, b_val, b_test = split_patients(benign_shuffled, TRAIN_RATIO, VAL_RATIO)
m_train, m_val, m_test = split_patients(malignant_shuffled, TRAIN_RATIO, VAL_RATIO)

train_patients = b_train + m_train
val_patients = b_val + m_val
test_patients = b_test + m_test

print(f"  Benign:    {len(b_train)} train, {len(b_val)} val, {len(b_test)} test")
print(f"  Malignant: {len(m_train)} train, {len(m_val)} val, {len(m_test)} test")
print(f"  Total:     {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test")

# -------------------------------------------------------------------------
# Step 3: Create symlinked directory structure
# -------------------------------------------------------------------------
print(f"\nStep 3: Creating directory structure with symlinks...")

data_dir = os.path.join(EXPERIMENT_BASE, "data")

for split_name, split_patients_list in [("train", train_patients),
                                         ("validation", val_patients),
                                         ("test", test_patients)]:
    split_crops = 0
    for cls in ["benign", "malignant"]:
        out_dir = os.path.join(data_dir, split_name, cls)
        os.makedirs(out_dir, exist_ok=True)

        for pid in split_patients_list:
            for crop_path, crop_cls in patient_crops[pid]:
                if crop_cls == cls:
                    link_path = os.path.join(out_dir, os.path.basename(crop_path))
                    if not os.path.exists(link_path):
                        os.symlink(crop_path, link_path)
                    split_crops += 1

    print(f"  {split_name}: {split_crops} crops")

# -------------------------------------------------------------------------
# Step 4: Run classifier — real-only and real+synthetic
# -------------------------------------------------------------------------
print(f"\nStep 4: Running classifier...")

start_time = time.time()

for mode, use_synthetic in [("real-only", False), ("real-and-synthetic", True)]:
    outdir = os.path.join(EXPERIMENT_BASE, mode)
    results_file = os.path.join(outdir, "results.json")

    if os.path.exists(results_file):
        print(f"  {mode}: SKIP (already done)")
        continue

    elapsed = time.time() - start_time
    print(f"  {mode}: running... (elapsed: {elapsed/60:.1f}m)")

    cmd = [
        "python", "project/pipeline/03-classifier-experiments/train_classifier.py",
        "--data-dir", data_dir,
        "--outdir", outdir,
    ]
    if use_synthetic:
        cmd.extend(["--synthetic-dir", SYNTHETIC_DIR,
                     "--synth-malignant", str(args.synth_malignant)])

    subprocess.run(cmd, check=True)

# -------------------------------------------------------------------------
# Step 5: Compare results
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

for mode in ["real-only", "real-and-synthetic"]:
    results_file = os.path.join(EXPERIMENT_BASE, mode, "results.json")
    if not os.path.exists(results_file):
        continue

    with open(results_file) as f:
        res = json.load(f)

    report = res["classification_report"]
    counts = res.get("image_counts", {})
    print(f"\n{mode}:")
    print(f"  Test accuracy:    {res['test_accuracy']:.4f}")
    print(f"  Benign recall:    {report['benign']['recall']:.4f}")
    print(f"  Malignant recall: {report['malignant']['recall']:.4f}")
    print(f"  Macro F1:         {report['macro avg']['f1-score']:.4f}")
    print(f"  Best val acc:     {res['best_val_accuracy']:.4f}")
    if counts:
        print(f"  Training images:  {counts.get('total_train', '?')} "
              f"({counts.get('real_benign', '?')}rb + "
              f"{counts.get('real_malignant', '?')}rm + "
              f"{counts.get('synth_benign', '?')}sb + "
              f"{counts.get('synth_malignant', '?')}sm)")

# Save experiment config
config = {
    "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
    "seed": SEED,
    "synth_malignant": args.synth_malignant,
    "train_patients": train_patients,
    "val_patients": val_patients,
    "test_patients": test_patients,
}
config_path = os.path.join(EXPERIMENT_BASE, "experiment_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

total_time = time.time() - start_time
print(f"\nTotal time: {total_time/60:.1f}m")
print(f"Config saved to {config_path}")
