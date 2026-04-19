# How to run (from the project root):
#   python project/pipeline/01-data-preprocessing/analysis/count_unique_dash_tokens.py path/to/your_file.txt FROM TO
#
# Dash boundary rules:
#   - FROM can be 0 to mean "start of line"
#   - TO is the dash number (1-based) and must be > FROM
#
# Examples:
#   # token between 2nd and 3rd dashes
#   python project/pipeline/01-data-preprocessing/analysis/count_unique_dash_tokens.py path/to/your_file.txt 2 3
#
#   # token between start of line (0th boundary) and 3rd dash
#   python project/pipeline/01-data-preprocessing/analysis/count_unique_dash_tokens.py path/to/your_file.txt 0 3
#
#   # token between 1st and 2nd dashes
#   python project/pipeline/01-data-preprocessing/analysis/count_unique_dash_tokens.py path/to/your_file.txt 1 2
#
# Example:
#   python project/pipeline/01-data-preprocessing/analysis/count_unique_dash_tokens.py project/data/mkfold/dsfold1.txt 2 3
#
# Output format:
#   Line 1: number of unique tokens in the chosen dash range
#   Line 2: one example in the format:
#           For row {x}, the token is {y}
#   Line 3: the full content of that example row

import argparse

def dash_positions(s: str, n: int):
    pos = []
    start = 0
    while len(pos) < n:
        i = s.find("-", start)
        if i == -1:
            break
        pos.append(i)
        start = i + 1
    return pos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("from_dash", type=int)  # 0 = start of line
    ap.add_argument("to_dash", type=int)    # 1.. and > from_dash
    args = ap.parse_args()

    if args.from_dash < 0 or args.to_dash < 1 or args.to_dash <= args.from_dash:
        raise SystemExit("Error: need from_dash >= 0, to_dash >= 1, and to_dash > from_dash")

    needed = args.to_dash
    tokens = set()
    skipped = 0

    example_row = None
    example_token = None
    example_line = None

    with open(args.file, "r", encoding="utf-8") as f:
        for row, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue

            pos = dash_positions(s, needed)
            if len(pos) < needed:
                skipped += 1
                continue

            left = -1 if args.from_dash == 0 else pos[args.from_dash - 1]
            right = pos[args.to_dash - 1]
            token = s[left + 1 : right]

            tokens.add(token)

            # store exactly one example (first valid one)
            if example_row is None:
                example_row = row
                example_token = token
                example_line = s  # store the row contents (stripped)

    print(f"{len(tokens)} unique tokens")
    if example_row is not None:
        print(f"As example, for row {example_row}, the token is {example_token}")
        print(f"Row {example_row} content: {example_line}")
    if skipped:
        print(f"(skipped {skipped} lines missing enough dashes)")

if __name__ == "__main__":
    main()