# How to run (from the project root):
#   python project/pipeline/01-data-preprocessing/01_trim_and_sort_fold.py path/to/dsfold_file.txt path/to/output_file.txt
#
# Example:
#   python project/pipeline/01-data-preprocessing/01_trim_and_sort_fold.py project/data/mkfold/dsfold1.txt project/data/breakhis_all_images_sorted.txt
#
# What it does:
#   Reads a dsfold file, trims each row to keep only the first two
#   pipe-separated fields (filename and magnification), sorts all rows
#   alphabetically in ascending order, and writes the result to a new file.
#
# Sample input:
#   SOB_B_A-14-22549AB-100-001.png|100|1|train
#
# Sample output:
#   SOB_B_A-14-22549AB-100-001.png|100

import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python project/pipeline/01-data-preprocessing/01_trim_and_sort_fold.py path/to/dsfold_file.txt path/to/output_file.txt")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path, "r") as f:
        lines = f.read().splitlines()

    trimmed = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("|")
        trimmed.append("|".join(parts[:2]))

    trimmed.sort()

    with open(output_path, "w") as f:
        f.write("\n".join(trimmed) + "\n")

    print(f"Processed {len(trimmed)} rows from {input_path} -> {output_path}")

if __name__ == "__main__":
    main()
