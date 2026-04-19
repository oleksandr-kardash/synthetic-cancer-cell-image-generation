# How to run (from the project root):
#   python project/pipeline/01-data-preprocessing/06_generate_dataset_json.py path/to/organised_by_mag_256_crops
#
# Example:
#   python project/pipeline/01-data-preprocessing/06_generate_dataset_json.py project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops
#
# What it does:
#   For each magnification's train folder, generates a dataset.json file
#   that maps each image to a class label (benign=0, malignant=1).
#   This file is required by dataset_tool.py for conditional training.
#
#   Creates: {mag}X/train/dataset.json

import sys
import os
import json

CLASS_LABELS = {"benign": 0, "malignant": 1}

def main():
    if len(sys.argv) != 2:
        print("Usage: python project/pipeline/01-data-preprocessing/06_generate_dataset_json.py path/to/organised_by_mag_256_crops")
        sys.exit(1)

    root_dir = sys.argv[1]

    for mag in sorted(os.listdir(root_dir), key=lambda x: int(x.rstrip("X"))):
        train_dir = os.path.join(root_dir, mag, "train")
        if not os.path.isdir(train_dir):
            continue

        labels = []
        for cls in sorted(os.listdir(train_dir)):
            cls_dir = os.path.join(train_dir, cls)
            if not os.path.isdir(cls_dir) or cls not in CLASS_LABELS:
                continue

            label = CLASS_LABELS[cls]
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.lower().endswith(".png"):
                    continue
                labels.append([f"{cls}/{fname}", label])

        metadata = {"labels": labels}
        json_path = os.path.join(train_dir, "dataset.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        print(f"{mag}/train: {len(labels)} labels written to dataset.json")

if __name__ == "__main__":
    main()
