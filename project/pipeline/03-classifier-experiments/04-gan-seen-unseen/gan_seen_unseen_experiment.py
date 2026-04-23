"""
GAN-seen vs GAN-unseen evaluation.

Tests whether synthetic-augmentation gains depend on whether the test
patients were seen by the GAN during its training. The GAN was trained on
57 patients (GAN-seen) from the original 70/15/15 split. The remaining 24
patients (GAN-unseen) were never seen by the GAN.

This experiment isolates one variable: are the test patients GAN-seen or
GAN-unseen? Everything else is kept as similar as possible — both
experiments use GAN-seen patients for validation (model selection), and
both have matched test set sizes (24 patients each).

Experiment A (GAN-seen test):
  - Test: 24 GAN-seen patients
  - Val: GAN-seen patients only
  - Train: remaining GAN-seen + all GAN-unseen patients

Experiment B (GAN-unseen test):
  - Test: all 24 GAN-unseen patients
  - Val: GAN-seen patients only
  - Train: remaining GAN-seen patients

If synthetic augmentation generalises uniformly, both experiments should
show similar improvement. If the benefit depends on whether the test
patients were in the GAN's training pool, only Experiment A should
benefit.

Both experiments run real-only and real+synthetic, then compare.
Synthetic balancing is handled by train_classifier.py via --synth-malignant.

Steps:
  1. Identify GAN-seen vs GAN-unseen patients from --split-file
  2. Gather all crops from --data-dir and map to patient IDs
  3. Create stratified splits for both experiments
  4. Create symlinked directory structures for real data
  5. Run train_classifier.py: real-only and real+synthetic for each experiment
  6. Compare results

Usage (from project root):
    python project/pipeline/03-classifier-experiments/04-gan-seen-unseen/gan_seen_unseen_experiment.py \
        --outdir project/classifier-runs/40x/gan-seen-unseen-experiment \
        --data-dir project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops/40X \
        --split-file project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15.txt \
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
    description="GAN-seen vs GAN-unseen evaluation: compares synthetic-augmentation "
                "gains when test patients were or were not part of the GAN's training pool"
)
parser.add_argument("--outdir", required=True,
                    help="Output directory for experiment results")
parser.add_argument("--data-dir", required=True,
                    help="Path to the 256x256 crop images organised by split "
                         "and class. All crops from all splits are combined and "
                         "re-split based on GAN-seen/unseen status.")
parser.add_argument("--split-file", required=True,
                    help="Path to the original patient split file. Patients with "
                         "split=train are GAN-seen (GAN trained on their images).")
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
SPLIT_FILE = args.split_file
SYNTHETIC_DIR = args.synthetic_dir
EXPERIMENT_BASE = args.outdir
SEED = args.seed

os.makedirs(EXPERIMENT_BASE, exist_ok=True)
sys.stdout = TeeOutput(os.path.join(EXPERIMENT_BASE, "log.txt"))


def get_patient_id(filename):
    """Extract patient ID using first three dash-separated tokens.
    e.g. SOB_B_A-14-22549CD-40-001-crop1.png -> SOB_B_A-14-22549CD
    Matches the original patient_split.py logic."""
    parts = filename.split("-")
    return "-".join(parts[:3])


# -------------------------------------------------------------------------
# Step 1: Identify GAN-seen and GAN-unseen patients
# -------------------------------------------------------------------------
print("Step 1: Identifying GAN-seen and GAN-unseen patients...")

gan_seen = set()
gan_unseen = set()

with open(SPLIT_FILE) as f:
    for line in f:
        parts = line.strip().split("|")
        fname = parts[0]
        split = parts[2]
        pid = get_patient_id(fname)
        if split == "train":
            gan_seen.add(pid)
        else:
            gan_unseen.add(pid)

print(f"  GAN-seen patients (original train): {len(gan_seen)}")
print(f"  GAN-unseen patients (original val+test): {len(gan_unseen)}")

# -------------------------------------------------------------------------
# Step 2: Gather all crops and map to patients
# -------------------------------------------------------------------------
print("\nStep 2: Gathering all crops...")

patient_crops = defaultdict(list)
patient_class = {}

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

print(f"  Total patients: {len(patient_class)}")
print(f"  Total crops: {sum(len(v) for v in patient_crops.values())}")

# -------------------------------------------------------------------------
# Step 3: Create splits for both experiments
# -------------------------------------------------------------------------
print("\nStep 3: Creating experiment splits...")

np.random.seed(SEED)

gan_seen_benign = sorted([p for p in gan_seen if patient_class[p] == "benign"])
gan_seen_malignant = sorted([p for p in gan_seen if patient_class[p] == "malignant"])
gan_unseen_benign = sorted([p for p in gan_unseen if patient_class[p] == "benign"])
gan_unseen_malignant = sorted([p for p in gan_unseen if patient_class[p] == "malignant"])

np.random.shuffle(gan_seen_benign)
np.random.shuffle(gan_seen_malignant)

print(f"  GAN-seen: {len(gan_seen_benign)} benign, {len(gan_seen_malignant)} malignant")
print(f"  GAN-unseen: {len(gan_unseen_benign)} benign, {len(gan_unseen_malignant)} malignant")

# --- Shared validation set ---
# Both experiments use the same GAN-seen patients for validation, so that
# model selection is identical and does not confound the comparison.
# Take ~20% of GAN-seen patients for val: floor(17*0.2)=3 benign,
# floor(40*0.2)=8 malignant = 11 patients.

n_val_ben = int(np.floor(len(gan_seen_benign) * 0.2))   # 3
n_val_mal = int(np.floor(len(gan_seen_malignant) * 0.2)) # 8

val_ben = gan_seen_benign[:n_val_ben]
val_mal = gan_seen_malignant[:n_val_mal]
val_patients = val_ben + val_mal

# Remaining GAN-seen after removing val: 14 benign + 32 malignant = 46
remaining_seen_ben = gan_seen_benign[n_val_ben:]
remaining_seen_mal = gan_seen_malignant[n_val_mal:]

print(f"\n  Shared val: {len(val_patients)} patients ({len(val_ben)} benign, {len(val_mal)} malignant) — all GAN-seen")
print(f"  Remaining GAN-seen: {len(remaining_seen_ben)} benign, {len(remaining_seen_mal)} malignant")

# --- Experiment A: GAN-seen test ---
# Test: 24 GAN-seen patients from the remaining 46 (matched to Exp B's 24).
# Class proportions match GAN-unseen: 7 benign + 17 malignant.
# Train: remaining 22 GAN-seen + all 24 GAN-unseen = 46 patients.

n_a_test_ben = len(gan_unseen_benign)    # 7, matching Exp B test composition
n_a_test_mal = len(gan_unseen_malignant) # 17

a_test_ben = remaining_seen_ben[:n_a_test_ben]
a_test_mal = remaining_seen_mal[:n_a_test_mal]
a_train_ben = remaining_seen_ben[n_a_test_ben:] + list(gan_unseen_benign)
a_train_mal = remaining_seen_mal[n_a_test_mal:] + list(gan_unseen_malignant)

exp_a = {
    "train": a_train_ben + a_train_mal,
    "validation": val_patients,
    "test": a_test_ben + a_test_mal,
}

print(f"\n  Experiment A (GAN-seen test):")
print(f"    Test:  {len(exp_a['test'])} patients ({len(a_test_ben)} benign, {len(a_test_mal)} malignant) — all GAN-seen")
print(f"    Val:   {len(exp_a['validation'])} patients — shared")
print(f"    Train: {len(exp_a['train'])} patients ({len(a_train_ben)} benign, {len(a_train_mal)} malignant)")

# --- Experiment B: GAN-unseen test ---
# Test: all 24 GAN-unseen patients (7 benign + 17 malignant).
# Train: all 46 remaining GAN-seen patients.
# Val: same shared val as Experiment A.

b_train_ben = list(remaining_seen_ben)
b_train_mal = list(remaining_seen_mal)

exp_b = {
    "train": b_train_ben + b_train_mal,
    "validation": val_patients,
    "test": list(gan_unseen_benign) + list(gan_unseen_malignant),
}

print(f"\n  Experiment B (GAN-unseen test):")
print(f"    Test:  {len(exp_b['test'])} patients ({len(gan_unseen_benign)} benign, {len(gan_unseen_malignant)} malignant) — all GAN-unseen")
print(f"    Val:   {len(exp_b['validation'])} patients — shared")
print(f"    Train: {len(exp_b['train'])} patients ({len(b_train_ben)} benign, {len(b_train_mal)} malignant)")

# -------------------------------------------------------------------------
# Step 4: Create symlinked directory structures
# -------------------------------------------------------------------------
print("\nStep 4: Creating directory structures...")

experiments = {"gan-seen-test": exp_a, "gan-unseen-test": exp_b}

for exp_name, splits in experiments.items():
    data_dir = os.path.join(EXPERIMENT_BASE, exp_name, "data")

    for split_name, split_patients in splits.items():
        for cls in ["benign", "malignant"]:
            out_dir = os.path.join(data_dir, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)

            for pid in split_patients:
                for crop_path, crop_cls in patient_crops[pid]:
                    if crop_cls == cls:
                        link_path = os.path.join(out_dir, os.path.basename(crop_path))
                        if not os.path.exists(link_path):
                            os.symlink(crop_path, link_path)

    for split_name in ["train", "validation", "test"]:
        ben = len(os.listdir(os.path.join(data_dir, split_name, "benign")))
        mal = len(os.listdir(os.path.join(data_dir, split_name, "malignant")))
        print(f"  {exp_name} {split_name}: {ben} benign + {mal} malignant = {ben + mal} crops")

# -------------------------------------------------------------------------
# Step 5: Run classifier on both experiments
# -------------------------------------------------------------------------
print("\nStep 5: Running classifier...")

start_time = time.time()
all_results = {}

for exp_name in ["gan-seen-test", "gan-unseen-test"]:
    data_dir = os.path.join(EXPERIMENT_BASE, exp_name, "data")
    all_results[exp_name] = {}

    for mode, use_synthetic in [("real-only", False), ("real-and-synthetic", True)]:
        outdir = os.path.join(EXPERIMENT_BASE, exp_name, mode)
        results_file = os.path.join(outdir, "results.json")

        if os.path.exists(results_file):
            print(f"  {exp_name}/{mode}: SKIP (already done)")
            with open(results_file) as f:
                all_results[exp_name][mode] = json.load(f)
            continue

        elapsed = time.time() - start_time
        print(f"  {exp_name}/{mode}: running... (elapsed: {elapsed/60:.1f}m)")

        cmd = [
            "python", "project/pipeline/03-classifier-experiments/train_classifier.py",
            "--data-dir", data_dir,
            "--outdir", outdir,
        ]
        if use_synthetic:
            cmd.extend(["--synthetic-dir", SYNTHETIC_DIR,
                         "--synth-malignant", str(args.synth_malignant)])

        subprocess.run(cmd, check=True)

        with open(results_file) as f:
            all_results[exp_name][mode] = json.load(f)

# -------------------------------------------------------------------------
# Step 6: Compare results
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GAN-SEEN VS GAN-UNSEEN EVALUATION RESULTS")
print("=" * 70)

for exp_name, label in [("gan-seen-test", "Experiment A: GAN-SEEN test patients"),
                          ("gan-unseen-test", "Experiment B: GAN-UNSEEN test patients")]:
    print(f"\n{label}")
    print("-" * 50)

    for mode in ["real-only", "real-and-synthetic"]:
        r = all_results[exp_name][mode]
        report = r["classification_report"]
        counts = r.get("image_counts", {})
        print(f"  {mode}:")
        print(f"    Test accuracy:    {r['test_accuracy']:.4f}")
        print(f"    Benign recall:    {report['benign']['recall']:.4f}")
        print(f"    Malignant recall: {report['malignant']['recall']:.4f}")
        print(f"    Macro F1:         {report['macro avg']['f1-score']:.4f}")
        if counts:
            print(f"    Training images:  {counts.get('total_train', '?')} "
                  f"({counts.get('real_benign', '?')}rb + "
                  f"{counts.get('real_malignant', '?')}rm + "
                  f"{counts.get('synth_benign', '?')}sb + "
                  f"{counts.get('synth_malignant', '?')}sm)")

# Summary comparison
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"{'':30s} {'Real Only':>12s} {'Real+Synth':>12s} {'Diff':>8s}")
print("-" * 62)

for exp_name, label in [("gan-seen-test", "A: GAN-seen test"),
                          ("gan-unseen-test", "B: GAN-unseen test")]:
    real = all_results[exp_name]["real-only"]["test_accuracy"]
    synth = all_results[exp_name]["real-and-synthetic"]["test_accuracy"]
    diff = synth - real
    print(f"  {label:28s} {real:11.1%} {synth:11.1%} {diff:+7.1%}")

# Save config
config = {
    "seed": SEED,
    "synth_malignant": args.synth_malignant,
    "gan_seen_patients": sorted(gan_seen),
    "gan_unseen_patients": sorted(gan_unseen),
    "experiment_a": {k: v for k, v in exp_a.items()},
    "experiment_b": {k: v for k, v in exp_b.items()},
}
config_path = os.path.join(EXPERIMENT_BASE, "experiment_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

total_time = time.time() - start_time
print(f"\nTotal time: {total_time/60:.1f}m")
print(f"Config saved to {config_path}")
