# How to run (from the project root):
#   python project/pipeline/01-data-preprocessing/03_patient_split.py path/to/input_file.txt path/to/output_file.txt
#
# Example:
#   python project/pipeline/01-data-preprocessing/03_patient_split.py project/data/breakhis_no_SOB_M_DC-14-13412_sorted.txt project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15.txt
#
# What it does:
#   Reads a trimmed and sorted image list (output of trim_and_sort_fold.py),
#   assigns each image to train/validation/test based on a 70/15/15
#   stratified patient-wise split, and writes the result to a new file.
#   Stratified means benign and malignant patients are split independently,
#   guaranteeing both classes appear in every split.
#   A patient is defined by the first three dash-separated tokens of the
#   filename (e.g. SOB_B_A-14-22549AB). All images from the same patient
#   are assigned to the same split. Uses a fixed random seed for
#   reproducibility.
#
# Output format:
#   SOB_B_A-14-22549AB-100-001.png|100|train

import sys
import math
import random
from collections import OrderedDict

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (remainder)

def get_patient_id(line):
    filename = line.split("|")[0]
    parts = filename.split("-")
    return "-".join(parts[:3])

def main():
    if len(sys.argv) != 3:
        print("Usage: python project/pipeline/01-data-preprocessing/03_patient_split.py path/to/input_file.txt path/to/output_file.txt")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path, "r") as f:
        lines = f.read().splitlines()

    lines = [line for line in lines if line.strip()]

    # Group images by patient
    patients = OrderedDict()
    for line in lines:
        pid = get_patient_id(line)
        if pid not in patients:
            patients[pid] = []
        patients[pid].append(line)

    # Separate patients into benign and malignant
    benign_ids = [pid for pid in patients if pid.startswith("SOB_B_")]
    malignant_ids = [pid for pid in patients if pid.startswith("SOB_M_")]

    # Shuffle each group independently with fixed seed
    random.seed(SEED)
    random.shuffle(benign_ids)
    random.shuffle(malignant_ids)

    # Split each group into train/val/test
    def split_list(ids):
        n = len(ids)
        n_train = round(n * TRAIN_RATIO)
        n_val = math.floor(n * VAL_RATIO)
        return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]

    b_train, b_val, b_test = split_list(benign_ids)
    m_train, m_val, m_test = split_list(malignant_ids)

    train_patients = set(b_train + m_train)
    val_patients = set(b_val + m_val)
    test_patients = set(b_test + m_test)

    print(f"  Benign patients:    {len(b_train)} train, {len(b_val)} validation, {len(b_test)} test")
    print(f"  Malignant patients: {len(m_train)} train, {len(m_val)} validation, {len(m_test)} test")

    # Assign splits and write output
    output_lines = []
    for line in lines:
        pid = get_patient_id(line)
        if pid in train_patients:
            split = "train"
        elif pid in val_patients:
            split = "validation"
        else:
            split = "test"
        output_lines.append(f"{line}|{split}")

    with open(output_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")

    total_patients = len(benign_ids) + len(malignant_ids)
    print(f"Split {total_patients} patients ({len(lines)} images) -> {output_path}")
    print(f"  Patients: {len(train_patients)} train, {len(val_patients)} validation, {len(test_patients)} test")

if __name__ == "__main__":
    main()
