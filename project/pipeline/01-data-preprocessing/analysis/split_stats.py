# How to run (from the project root):
#   python project/pipeline/01-data-preprocessing/analysis/split_stats.py path/to/split_file.txt
#
# Example:
#   python project/pipeline/01-data-preprocessing/analysis/split_stats.py project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15.txt
#
# What it does:
#   Reads a split file (output of patient_split.py) and prints counts and
#   proportions of train/validation/test images for each magnification
#   (40x, 100x, 200x, 400x) and overall.
#
# Output format (expected input):
#   SOB_B_A-14-22549AB-100-001.png|100|train

import sys
from collections import defaultdict

def main():
    if len(sys.argv) != 2:
        print("Usage: python project/pipeline/01-data-preprocessing/analysis/split_stats.py path/to/split_file.txt")
        sys.exit(1)

    input_path = sys.argv[1]

    with open(input_path, "r") as f:
        lines = f.read().splitlines()

    lines = [line for line in lines if line.strip()]

    # Count per magnification, class, and split
    counts = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)
    cls_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    cls_total = defaultdict(lambda: defaultdict(int))

    for line in lines:
        parts = line.split("|")
        imgname = parts[0]
        mag = parts[1]
        split = parts[2]
        cls = "benign" if imgname.startswith("SOB_B_") else "malignant"
        counts[mag][split] += 1
        total[split] += 1
        cls_counts[mag][cls][split] += 1
        cls_total[cls][split] += 1

    magnifications = sorted(counts.keys(), key=lambda x: int(x))
    splits = ["train", "validation", "test"]
    classes = ["benign", "malignant"]

    def print_row(label, split_counts):
        row_total = sum(split_counts.get(s, 0) for s in splits)
        parts = []
        for s in splits:
            c = split_counts.get(s, 0)
            pct = c / row_total * 100 if row_total > 0 else 0
            parts.append(f"{c:>5d} ({pct:4.1f}%)")
        print(f"{label:>20s}  {parts[0]}  {parts[1]}  {parts[2]}  {row_total:>8d}")

    # Print header
    print(f"{'':>20s}  {'train':>12s}  {'validation':>12s}  {'test':>12s}  {'total':>8s}")
    print("-" * 76)

    for mag in magnifications:
        print_row(mag + "x", counts[mag])
        for cls in classes:
            print_row(f"  {cls}", cls_counts[mag][cls])

    # Overall
    print("-" * 76)
    print_row("Overall", total)
    for cls in classes:
        print_row(f"  {cls}", cls_total[cls])

if __name__ == "__main__":
    main()
