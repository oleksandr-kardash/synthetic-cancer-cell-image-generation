# How to run (from the project root):
#   python project/pipeline/01-data-preprocessing/02_remove_rows_by_prefix.py path/to/input_file.txt path/to/output_file.txt PREFIX
#
# Example:
#   python project/pipeline/01-data-preprocessing/02_remove_rows_by_prefix.py project/data/breakhis_all_images_sorted.txt project/data/breakhis_no_SOB_M_DC-14-13412_sorted.txt SOB_M_DC-14-13412
#
# What it does:
#   Reads a text file, removes all rows that start with the given prefix,
#   and writes the remaining rows to a new file.

import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python project/pipeline/01-data-preprocessing/02_remove_rows_by_prefix.py path/to/input_file.txt path/to/output_file.txt PREFIX")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    prefix = sys.argv[3]

    with open(input_path, "r") as f:
        lines = f.read().splitlines()

    original_count = len(lines)
    filtered = [line for line in lines if line.strip() and not line.startswith(prefix)]

    with open(output_path, "w") as f:
        f.write("\n".join(filtered) + "\n")

    removed = original_count - len(filtered)
    print(f"Removed {removed} rows starting with '{prefix}'")
    print(f"{len(filtered)} rows remaining -> {output_path}")

if __name__ == "__main__":
    main()
