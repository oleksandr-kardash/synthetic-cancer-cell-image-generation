# How to run (from the project root):
#   python project/pipeline/01-data-preprocessing/04_copy_breakhis_images_into_magnification_split_folders.py path/to/BreaKHis_v1 path/to/split_file.txt path/to/output_dir
#
# Example:
#   python project/pipeline/01-data-preprocessing/04_copy_breakhis_images_into_magnification_split_folders.py project/data/BreaKHis_v1 project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15.txt project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag
#
# What it does:
#   Reads a split file (output of patient_split.py) and copies images from the
#   BreaKHis_v1 source directory into a folder structure organised by
#   magnification first, then split:
#
#     output_dir/
#       40X/train/benign/   40X/train/malignant/
#       40X/validation/benign/  40X/validation/malignant/
#       40X/test/benign/    40X/test/malignant/
#       (same for 100X, 200X, 400X)

import sys
import os
import shutil

SRCFILES = {
    'DC': '%s/malignant/SOB/ductal_carcinoma/%s/%sX/%s',
    'LC': '%s/malignant/SOB/lobular_carcinoma/%s/%sX/%s',
    'MC': '%s/malignant/SOB/mucinous_carcinoma/%s/%sX/%s',
    'PC': '%s/malignant/SOB/papillary_carcinoma/%s/%sX/%s',
    'A':  '%s/benign/SOB/adenosis/%s/%sX/%s',
    'F':  '%s/benign/SOB/fibroadenoma/%s/%sX/%s',
    'PT': '%s/benign/SOB/phyllodes_tumor/%s/%sX/%s',
    'TA': '%s/benign/SOB/tubular_adenoma/%s/%sX/%s',
}

def get_src_path(root_dir, imgname, mag):
    tumor = imgname.split('-')[0].split('_')[-1]
    s = imgname.split('-')
    sub = s[0] + '_' + s[1] + '-' + s[2]
    return SRCFILES[tumor] % (root_dir, sub, mag, imgname)

def main():
    if len(sys.argv) != 4:
        print("Usage: python project/pipeline/01-data-preprocessing/04_copy_breakhis_images_into_magnification_split_folders.py path/to/BreaKHis_v1 path/to/split_file.txt path/to/output_dir")
        sys.exit(1)

    breakhis_dir = sys.argv[1]
    input_path = sys.argv[2]
    output_dir = sys.argv[3]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    root_dir = os.path.join(breakhis_dir, 'histology_slides', 'breast')

    with open(input_path, "r") as f:
        lines = f.read().splitlines()

    lines = [line for line in lines if line.strip()]

    copied = 0
    errors = 0

    for line in lines:
        parts = line.split("|")
        imgname = parts[0]
        mag = parts[1]
        split = parts[2]

        cls = "benign" if imgname.startswith("SOB_B_") else "malignant"
        dst_dir = os.path.join(output_dir, mag + "X", split, cls)
        os.makedirs(dst_dir, exist_ok=True)

        src_path = get_src_path(root_dir, imgname, mag)
        dst_path = os.path.join(dst_dir, imgname)

        if not os.path.exists(src_path):
            print(f"WARNING: source not found: {src_path}")
            errors += 1
            continue

        shutil.copy2(src_path, dst_path)
        copied += 1

    print(f"Copied {copied} images to {output_dir}")
    if errors > 0:
        print(f"WARNING: {errors} source files not found")

if __name__ == "__main__":
    main()
