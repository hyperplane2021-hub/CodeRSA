import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize baseline_pairwise_avg.csv across seed run roots.")
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--out", default="/workspace/seed_summary.csv")
    args = parser.parse_args()

    rows = []
    for root_text in args.roots:
        root = Path(root_text)
        csv_path = root / "baseline_pairwise_avg.csv"
        config_path = root / "run_config.json"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        seed = root.name
        if config_path.exists():
            payload = json.loads(config_path.read_text())
            seed = payload.get("config", payload).get("seed", seed)
        df = pd.read_csv(csv_path)
        row = {"seed": seed, "root": str(root)}
        row.update(dict(zip(df["method"], df["accuracy"])))
        rows.append(row)

    per_seed = pd.DataFrame(rows).sort_values("seed")
    metric_cols = [col for col in per_seed.columns if col not in {"seed", "root"}]
    summary = per_seed[metric_cols].agg(["mean", "std"]).reset_index().rename(columns={"index": "stat"})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out_path.with_name(out_path.stem + "_per_seed.csv"), index=False)
    summary.to_csv(out_path, index=False)
    print("per-seed:")
    print(per_seed.to_string(index=False))
    print("\nsummary:")
    print(summary.to_string(index=False))
    print("\nsaved:", out_path)
    print("saved:", out_path.with_name(out_path.stem + "_per_seed.csv"))


if __name__ == "__main__":
    main()
