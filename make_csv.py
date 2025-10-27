#!/usr/bin/env python
# scripts/make_csv.py
"""
Scan a Libri2Mix split and write a CSV with columns:
mix_path,s1_path,s2_path
"""
import argparse, csv, pathlib, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="Split folder, e.g. $LIBRI2MIX_ROOT/train-100-clean")
    p.add_argument("--out",  required=True,
                   help="Output CSV path, e.g. data/train.csv")
    args = p.parse_args()

    root = pathlib.Path(args.root).expanduser().resolve()
    mix_dir = root / "mix"
    s1_dir  = root / "s1"
    s2_dir  = root / "s2"

    assert mix_dir.is_dir(), f"Missing {mix_dir}"
    rows = []
    for mix_path in mix_dir.glob("*.wav"):
        stem = mix_path.stem            # e.g. 84_121550_84_121552
        s1_path = s1_dir / f"{stem}.wav"
        s2_path = s2_dir / f"{stem}.wav"
        if not (s1_path.exists() and s2_path.exists()):
            print(f"[warn] pair missing for {stem}", file=sys.stderr)
            continue
        rows.append((str(mix_path), str(s1_path), str(s2_path)))

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["mix_path", "s1_path", "s2_path"])
        wr.writerows(rows)

    print(f"Wrote {len(rows)} rows → {out_path}")

if __name__ == "__main__":
    main()
