"""
K-fold patient-level stratified cross-validation for BreaKHis classifier.

Uses the PyTorch tutorial baseline (train_classifier.py) with no
hyperparameter modifications. Runs each fold real-only and real+synthetic,
then reports mean +/- std across folds.

Also analyses GAN data leakage: for each fold, reports how many test and
val patients were in the original GAN training set, and correlates this
with the synthetic data improvement per fold.

Steps:
  1. Pool crops from --data-dir (all splits: train/validation/test)
  2. Map each crop to its patient ID
  3. Identify GAN-seen vs GAN-unseen patients from --split-file
  4. Create K stratified folds (balanced benign/malignant)
  5. For each fold: test=fold[i], val=fold[i+1], train=remaining K-2 folds
  6. Create symlinked directory structures for real data
  7. Run train_classifier.py on each fold (real-only and real+synthetic)
     Synthetic balancing is handled by train_classifier.py via --synth-malignant
  8. Aggregate results and analyse GAN leakage

Usage (from project root):
    python project/pipeline/03-classifier-experiments/03-cross-validation/cross_validation.py \
        --outdir project/classifier-runs/40x \
        --data-dir project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops/40X \
        --split-file project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15.txt \
        --synthetic-dir project/synthetic-images/40x-snapshot-008800

    Output will be saved to: project/classifier-runs/40x/cv-5fold-seed42-0-malignant/
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
    description="5-fold patient-level stratified cross-validation for BreaKHis classifier"
)
parser.add_argument("--outdir", required=True,
                    help="Base output directory (e.g. project/classifier-runs/40x). "
                         "A subfolder named cv-{folds}fold-seed{seed} will be "
                         "created inside this directory.")
parser.add_argument("--data-dir", required=True,
                    help="Path to the 256x256 crop images organised by split "
                         "and class. Expected structure: "
                         "{data-dir}/{train,validation,test}/{benign,malignant}/*.png. "
                         "All crops from all splits are combined and "
                         "re-split into CV folds at the patient level.")
parser.add_argument("--split-file", required=True,
                    help="Path to the original patient split file that was used "
                         "to define the GAN's training set. Each line has the "
                         "format: filename|magnification|split. Patients with "
                         "split=train are considered 'GAN-seen' (the GAN was "
                         "trained on their images). This is used for the data "
                         "leakage analysis — it does NOT affect fold assignment.")
parser.add_argument("--synthetic-dir", default=None,
                    help="Path to GAN-generated synthetic images with class "
                         "subfolders (e.g. {synthetic-dir}/{benign,malignant}/*.png). "
                         "If provided, each fold runs both real-only and "
                         "real+synthetic. If omitted, only real-only runs.")
parser.add_argument("--synth-malignant", type=int, default=0,
                    help="Number of synthetic malignant images to add per fold. "
                         "The number of synthetic benign images is then computed "
                         "automatically to balance the classes: "
                         "real_benign + synth_benign = real_malignant + synth_malignant. "
                         "Default: 0 (only synthetic benign added to match "
                         "real malignant count).")
parser.add_argument("--folds", type=int, default=5,
                    help="Number of CV folds (default: 5)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for patient shuffling before fold "
                         "assignment. Same seed = same folds (default: 42)")
args = parser.parse_args()

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
CROP_BASE = args.data_dir
SPLIT_FILE = args.split_file
SYNTHETIC_DIR = args.synthetic_dir
CV_BASE = os.path.join(args.outdir, f"cv-{args.folds}fold-seed{args.seed}-{args.synth_malignant}-malignant")
K = args.folds
SEED = args.seed

os.makedirs(CV_BASE, exist_ok=True)
sys.stdout = TeeOutput(os.path.join(CV_BASE, "log.txt"))
print(f"Output directory: {CV_BASE}")

# -------------------------------------------------------------------------
# Step 1: Gather all crops and map to patients
# -------------------------------------------------------------------------
print("Step 1: Gathering all crops and mapping to patients...")

patient_crops = defaultdict(list)   # patient_id -> [(abs_crop_path, class)]
patient_class = {}                  # patient_id -> "benign" or "malignant"

def get_patient_id(filename):
    """Extract patient ID using first three dash-separated tokens.
    e.g. SOB_B_A-14-22549CD-40-001-crop1.png -> SOB_B_A-14-22549CD
    Matches the original patient_split.py logic."""
    parts = filename.split("-")
    return "-".join(parts[:3])


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
# Step 1b: Identify GAN-seen vs GAN-unseen patients
# -------------------------------------------------------------------------
# The GAN was trained on the original 70/15/15 split's training patients.
# Patients in the original val/test sets were never seen by the GAN.
# -------------------------------------------------------------------------
print("\nStep 1b: Identifying GAN-seen vs GAN-unseen patients...")

gan_seen = set()
with open(SPLIT_FILE) as f:
    for line in f:
        parts = line.strip().split("|")
        fname = parts[0]
        split = parts[2]
        pid = get_patient_id(fname)
        if split == "train":
            gan_seen.add(pid)

gan_unseen = set(patient_class.keys()) - gan_seen
print(f"  GAN-seen (original train): {len(gan_seen)}")
print(f"  GAN-unseen (original val+test): {len(gan_unseen)}")

# -------------------------------------------------------------------------
# Step 2: Create 5 stratified folds
# -------------------------------------------------------------------------
print(f"\nStep 2: Creating {K} stratified folds (seed={SEED})...")

np.random.seed(SEED)

benign_shuffled = benign_patients.copy()
malignant_shuffled = malignant_patients.copy()
np.random.shuffle(benign_shuffled)
np.random.shuffle(malignant_shuffled)


def split_into_folds(patients, k):
    """Distribute patients into k roughly equal folds."""
    folds = [[] for _ in range(k)]
    for i, p in enumerate(patients):
        folds[i % k].append(p)
    return folds


benign_folds = split_into_folds(benign_shuffled, K)
malignant_folds = split_into_folds(malignant_shuffled, K)
folds = [b + m for b, m in zip(benign_folds, malignant_folds)]

for i, fold in enumerate(folds):
    n_ben = sum(1 for p in fold if patient_class[p] == "benign")
    n_mal = sum(1 for p in fold if patient_class[p] == "malignant")
    n_crops = sum(len(patient_crops[p]) for p in fold)
    print(f"  Fold {i+1}: {len(fold)} patients ({n_ben} benign, {n_mal} malignant), {n_crops} crops")

# -------------------------------------------------------------------------
# Step 3: Create symlinked directory structures for each fold
# -------------------------------------------------------------------------
# Fold i: test=folds[i], val=folds[(i+1) % K], train=remaining 3 folds
# -------------------------------------------------------------------------
print(f"\nStep 3: Creating directory structures with symlinks...")

fold_data_dirs = []

for i in range(K):
    test_idx = i
    val_idx = (i + 1) % K
    train_idxs = [j for j in range(K) if j != test_idx and j != val_idx]

    test_patients = folds[test_idx]
    val_patients = folds[val_idx]
    train_patients = []
    for j in train_idxs:
        train_patients.extend(folds[j])

    fold_data_dir = os.path.join(CV_BASE, f"fold{i+1}", "data")
    fold_data_dirs.append(fold_data_dir)

    # Count crops per split
    splits_info = {}

    for split_name, split_patients in [("train", train_patients),
                                        ("validation", val_patients),
                                        ("test", test_patients)]:
        split_crops = 0
        for cls in ["benign", "malignant"]:
            out_dir = os.path.join(fold_data_dir, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)

            for pid in split_patients:
                for crop_path, crop_cls in patient_crops[pid]:
                    if crop_cls == cls:
                        link_path = os.path.join(out_dir, os.path.basename(crop_path))
                        if not os.path.exists(link_path):
                            os.symlink(crop_path, link_path)
                        split_crops += 1

        splits_info[split_name] = split_crops

    print(f"  Fold {i+1}: train={splits_info['train']} crops, "
          f"val={splits_info['validation']} crops, "
          f"test={splits_info['test']} crops")

# -------------------------------------------------------------------------
# Step 4: Run train_classifier.py on each fold
# -------------------------------------------------------------------------
# Balancing of synthetic data is handled by train_classifier.py itself
# via --synth-malignant. The full synthetic directory is passed directly.
# -------------------------------------------------------------------------
print(f"\nStep 4: Running classifier on each fold...")

all_results = {"real_only": [], "real_and_synthetic": []}
start_time = time.time()

for i in range(K):
    fold_data_dir = fold_data_dirs[i]

    modes = [("real_only", False)]
    if SYNTHETIC_DIR is not None:
        modes.append(("real_and_synthetic", True))
    for mode, synthetic in modes:
        outdir = os.path.join(CV_BASE, f"fold{i+1}", mode)
        results_file = os.path.join(outdir, "results.json")

        # Skip if already completed
        if os.path.exists(results_file):
            print(f"  Fold {i+1} {mode}: SKIP (already done)")
            with open(results_file) as f:
                res = json.load(f)
            all_results[mode].append(res)
            continue

        elapsed = time.time() - start_time
        print(f"  Fold {i+1} {mode}: running... (elapsed: {elapsed/60:.1f}m)")

        cmd = [
            "python", "project/pipeline/03-classifier-experiments/train_classifier.py",
            "--data-dir", fold_data_dir,
            "--outdir", outdir,
        ]
        if synthetic:
            cmd.extend(["--synthetic-dir", SYNTHETIC_DIR,
                         "--synth-malignant", str(args.synth_malignant)])

        try:
            subprocess.run(cmd, check=True)
            with open(results_file) as f:
                res = json.load(f)
            all_results[mode].append(res)
            print(f"    -> test_acc={res['test_accuracy']:.4f}")
        except subprocess.CalledProcessError as e:
            print(f"    -> FAILED")

# -------------------------------------------------------------------------
# Step 5: Aggregate and report results
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CROSS-VALIDATION RESULTS")
print("=" * 70)

summary = {}

for mode in ["real_only", "real_and_synthetic"]:
    results_list = all_results[mode]
    if not results_list:
        continue

    test_accs = [r["test_accuracy"] for r in results_list]
    benign_recalls = [r["classification_report"]["benign"]["recall"] for r in results_list]
    mal_recalls = [r["classification_report"]["malignant"]["recall"] for r in results_list]
    macro_f1s = [r["classification_report"]["macro avg"]["f1-score"] for r in results_list]

    label = "Real only" if mode == "real_only" else "Real + Synthetic"
    print(f"\n{label} ({len(results_list)} folds):")
    print(f"  Test accuracy:   {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")
    print(f"  Benign recall:   {np.mean(benign_recalls):.4f} +/- {np.std(benign_recalls):.4f}")
    print(f"  Malignant recall:{np.mean(mal_recalls):.4f} +/- {np.std(mal_recalls):.4f}")
    print(f"  Macro F1:        {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}")

    print(f"\n  Per-fold breakdown:")
    for j, r in enumerate(results_list):
        report = r["classification_report"]
        counts = r.get("image_counts", {})
        counts_str = ""
        if counts:
            counts_str = (f"  train={counts.get('total_train', '?')} "
                          f"({counts.get('real_benign', '?')}rb+"
                          f"{counts.get('real_malignant', '?')}rm+"
                          f"{counts.get('synth_benign', '?')}sb+"
                          f"{counts.get('synth_malignant', '?')}sm)")
        print(f"    Fold {j+1}: test_acc={r['test_accuracy']:.4f}  "
              f"benign_recall={report['benign']['recall']:.4f}  "
              f"mal_recall={report['malignant']['recall']:.4f}  "
              f"macro_f1={report['macro avg']['f1-score']:.4f}"
              f"{counts_str}")

    summary[mode] = {
        "test_accuracy_mean": float(np.mean(test_accs)),
        "test_accuracy_std": float(np.std(test_accs)),
        "benign_recall_mean": float(np.mean(benign_recalls)),
        "benign_recall_std": float(np.std(benign_recalls)),
        "malignant_recall_mean": float(np.mean(mal_recalls)),
        "malignant_recall_std": float(np.std(mal_recalls)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "per_fold": [
            {
                "test_accuracy": r["test_accuracy"],
                "benign_recall": r["classification_report"]["benign"]["recall"],
                "malignant_recall": r["classification_report"]["malignant"]["recall"],
                "macro_f1": r["classification_report"]["macro avg"]["f1-score"],
                "image_counts": r.get("image_counts", {}),
            }
            for r in results_list
        ],
    }

# -------------------------------------------------------------------------
# Step 6: GAN leakage analysis
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GAN DATA LEAKAGE ANALYSIS")
print("=" * 70)
print("\nFor each fold, how many test/val patients were in the GAN's training set:")
print(f"{'Fold':>4s}  {'Test GAN-seen':>13s}  {'Test unseen':>11s}  "
      f"{'Val GAN-seen':>12s}  {'Val unseen':>10s}  {'Synth improvement':>17s}")
print("-" * 75)

leakage_info = []

for i in range(K):
    test_idx = i
    val_idx = (i + 1) % K

    test_pats = set(folds[test_idx])
    val_pats = set(folds[val_idx])

    test_gan_seen = len(test_pats & gan_seen)
    test_gan_unseen = len(test_pats - gan_seen)
    val_gan_seen = len(val_pats & gan_seen)
    val_gan_unseen = len(val_pats - gan_seen)

    # Compute synthetic improvement if both results exist
    synth_diff = None
    if (len(all_results["real_only"]) > i and
            len(all_results["real_and_synthetic"]) > i):
        real_acc = all_results["real_only"][i]["test_accuracy"]
        synth_acc = all_results["real_and_synthetic"][i]["test_accuracy"]
        synth_diff = synth_acc - real_acc

    diff_str = f"{synth_diff:+.1%}" if synth_diff is not None else "N/A"

    print(f"  {i+1:>2d}  {test_gan_seen:>6d}/{len(test_pats):<6d}  "
          f"{test_gan_unseen:>5d}/{len(test_pats):<5d}  "
          f"{val_gan_seen:>5d}/{len(val_pats):<6d}  "
          f"{val_gan_unseen:>4d}/{len(val_pats):<5d}  "
          f"{diff_str:>11s}")

    fold_leakage = {
        "test_total": len(test_pats),
        "test_gan_seen": test_gan_seen,
        "test_gan_unseen": test_gan_unseen,
        "test_gan_seen_benign": len([p for p in test_pats & gan_seen if patient_class[p] == "benign"]),
        "test_gan_unseen_benign": len([p for p in test_pats - gan_seen if patient_class[p] == "benign"]),
        "val_total": len(val_pats),
        "val_gan_seen": val_gan_seen,
        "val_gan_unseen": val_gan_unseen,
        "synth_improvement": synth_diff,
    }
    leakage_info.append(fold_leakage)

# Add leakage info to summary
summary["leakage_analysis"] = leakage_info

# Save summary
summary_path = os.path.join(CV_BASE, "cv_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

# Save fold composition for reproducibility
fold_info = {
    f"fold{i+1}": {
        "patients": folds[i],
        "benign": [p for p in folds[i] if patient_class[p] == "benign"],
        "malignant": [p for p in folds[i] if patient_class[p] == "malignant"],
        "gan_seen": [p for p in folds[i] if p in gan_seen],
        "gan_unseen": [p for p in folds[i] if p in gan_unseen],
    }
    for i in range(K)
}
fold_info_path = os.path.join(CV_BASE, "fold_composition.json")
with open(fold_info_path, "w") as f:
    json.dump(fold_info, f, indent=2)

total_time = time.time() - start_time
print(f"\nTotal time: {total_time/60:.1f}m")
print(f"Summary saved to {summary_path}")
print(f"Fold composition saved to {fold_info_path}")
