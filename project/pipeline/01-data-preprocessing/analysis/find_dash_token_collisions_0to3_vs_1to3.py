# How to run (from the project root):
#   python project/pipeline/01-data-preprocessing/analysis/find_dash_token_collisions_0to3_vs_1to3.py path/to/your_file.txt
#
# Example:
#   python project/pipeline/01-data-preprocessing/analysis/find_dash_token_collisions_0to3_vs_1to3.py project/data/mkfold/dsfold1.txt
#
# Sample output:
#   Collision groups found: 1
#
#   ============================================================
#   [1] 1->3 token: 14-13412
#   Distinct 0->3 tokens: 2
#     - 0->3 token: SOB_M_DC-14-13412
#       Example row 1718: SOB_M_DC-14-13412-100-001.png|100|1|train
#     - 0->3 token: SOB_M_LC-14-13412
#       Example row 7152: SOB_M_LC-14-13412-100-001.png|100|1|test
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

def token_between(s: str, from_dash: int, to_dash: int) -> str | None:
    """
    Extract token between dash boundaries.
    from_dash=0 means start of string. to_dash is 1-based dash number.
    Example: from=1,to=3 => between 1st and 3rd dash.
    """
    if from_dash < 0 or to_dash < 1 or to_dash <= from_dash:
        return None

    pos = dash_positions(s, to_dash)
    if len(pos) < to_dash:
        return None

    left = -1 if from_dash == 0 else pos[from_dash - 1]
    right = pos[to_dash - 1]
    return s[left + 1 : right]

def main():
    ap = argparse.ArgumentParser(
        description="Find which 1->3 token corresponds to multiple 0->3 tokens."
    )
    ap.add_argument("file", help="Input text file")
    ap.add_argument("--max-print", type=int, default=50, help="Max collision groups to print")
    args = ap.parse_args()

    # Map: token_1_3 -> { token_0_3 -> (rownum, rowtext) }
    mapping = {}
    skipped = 0

    with open(args.file, "r", encoding="utf-8") as f:
        for row, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue

            t03 = token_between(s, 0, 3)  # start -> 3rd dash
            t13 = token_between(s, 1, 3)  # 1st -> 3rd dash

            if t03 is None or t13 is None:
                skipped += 1
                continue

            bucket = mapping.setdefault(t13, {})
            # store only one example row per distinct 0->3 token
            bucket.setdefault(t03, (row, s))

    # Find collisions: same 1->3 token mapping to >1 distinct 0->3 tokens
    collisions = [(t13, bucket) for t13, bucket in mapping.items() if len(bucket) > 1]
    collisions.sort(key=lambda x: len(x[1]), reverse=True)

    print(f"Collision groups found: {len(collisions)}")
    if skipped:
        print(f"(skipped {skipped} lines missing enough dashes)")

    for idx, (t13, bucket) in enumerate(collisions[: args.max_print], start=1):
        print("\n" + "=" * 60)
        print(f"[{idx}] 1->3 token: {t13}")
        print(f"Distinct 0->3 tokens: {len(bucket)}")

        for t03, (rownum, rowtext) in sorted(bucket.items(), key=lambda kv: kv[0]):
            print(f"  - 0->3 token: {t03}")
            print(f"    Example row {rownum}: {rowtext}")

if __name__ == "__main__":
    main()