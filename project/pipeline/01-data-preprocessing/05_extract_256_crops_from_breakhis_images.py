# How to run (from the project root):
#   python project/pipeline/01-data-preprocessing/05_extract_256_crops_from_breakhis_images.py path/to/organised_by_mag path/to/output_dir
#
# Example:
#   python project/pipeline/01-data-preprocessing/05_extract_256_crops_from_breakhis_images.py project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops
#
# What it does:
#   Reads BreaKHis images (700x460 or 700x456) from the organised folder
#   structure and extracts 256x256 crops at stride 222 (width) / 204 (height),
#   producing 6 crops per image with full coverage. For images shorter than
#   460px (e.g. 700x456), the second row is shifted up to fit, resulting in
#   slightly more vertical overlap.
#
#   Input structure:  {mag}X/{split}/{class}/{image}.png
#   Output structure: {mag}X/{split}/{class}/{image}-crop1.png ... -crop6.png

import sys
import os
import shutil
from PIL import Image

CROP_SIZE = 256
STRIDE_X = 222
STRIDE_Y = 204

def main():
    if len(sys.argv) != 3:
        print("Usage: python project/pipeline/01-data-preprocessing/05_extract_256_crops_from_breakhis_images.py path/to/organised_by_mag path/to/output_dir")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    total_crops = 0
    total_images = 0

    for mag in sorted(os.listdir(input_dir)):
        mag_dir = os.path.join(input_dir, mag)
        if not os.path.isdir(mag_dir):
            continue
        for split in sorted(os.listdir(mag_dir)):
            split_dir = os.path.join(mag_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for cls in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue

                out_cls_dir = os.path.join(output_dir, mag, split, cls)
                os.makedirs(out_cls_dir, exist_ok=True)

                for fname in sorted(os.listdir(cls_dir)):
                    if not fname.lower().endswith(".png"):
                        continue

                    src_path = os.path.join(cls_dir, fname)
                    img = Image.open(src_path)
                    w, h = img.size
                    base = fname[:-4]  # remove .png

                    # For 700x460: x=[0,222,444], y=[0,204] (stride 222/204)
                    # For 700x456: x=[0,222,444], y=[0,200] (stride 222/200, slightly more overlap)
                    x_positions = list(range(0, w - CROP_SIZE + 1, STRIDE_X))
                    y_positions = list(range(0, h - CROP_SIZE + 1, STRIDE_Y))
                    if y_positions[-1] + CROP_SIZE < h:
                        y_positions.append(h - CROP_SIZE)
                    if x_positions[-1] + CROP_SIZE < w:
                        x_positions.append(w - CROP_SIZE)

                    crop_num = 0
                    for x in x_positions:
                        for y in y_positions:
                            crop_num += 1
                            crop = img.crop((x, y, x + CROP_SIZE, y + CROP_SIZE))
                            crop_path = os.path.join(out_cls_dir, f"{base}-crop{crop_num}.png")
                            crop.save(crop_path)
                            total_crops += 1

                    total_images += 1

    print(f"Extracted {total_crops} crops from {total_images} images to {output_dir}")
    print(f"  Crop size: {CROP_SIZE}x{CROP_SIZE}, stride: {STRIDE_X}x{STRIDE_Y} (700x460) or {STRIDE_X}x200 (700x456)")

if __name__ == "__main__":
    main()
