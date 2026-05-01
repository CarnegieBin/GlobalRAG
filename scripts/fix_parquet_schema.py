"""
Fix schema inconsistencies across val parquet files so they can be concatenated
by HuggingFace datasets.concatenate_datasets().

Issues addressed:
1. reward_model.ground_truth.target: some files have Value('string'), others have
   Sequence(Value('string')) — normalize all to Sequence(Value('string'))
2. extra_info: some files have {'index', 'split'}, others have
   {'index', 'split', 'support_docs'} — normalize all to {'index', 'split'}
   (drop support_docs since it's Sequence(null) and not used in training)
3. golden_answers: check and normalize if needed

Usage (run on server):
    python scripts/fix_parquet_schema.py \
        --data_dir /ssd2/lini03/Search-R1-infer/test_model/data \
        --output_dir /ssd2/lini03/Search-R1-infer/test_model/data_fixed

Then update train script to point to data_fixed directory.
"""

import os
import argparse
import datasets
from datasets import Features, Value, Sequence


def normalize_parquet(input_path: str, output_path: str):
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")

    ds = datasets.load_dataset("parquet", data_files=input_path)["train"]
    print(f"  Original features:\n  {ds.features}")
    print(f"  Rows: {len(ds)}")

    changed = False

    # ------------------------------------------------------------------ #
    # Fix 1: reward_model.ground_truth.target -> Sequence(string)
    # ------------------------------------------------------------------ #
    rm_feature = ds.features.get("reward_model")
    if rm_feature is not None:
        # Determine the target field type
        if hasattr(rm_feature, "feature"):
            # datasets.Sequence wraps a dict feature
            gt_feature = rm_feature.feature.get("ground_truth", {})
        else:
            gt_feature = rm_feature.get("ground_truth", {})

        target_type = gt_feature.get("target") if isinstance(gt_feature, dict) else None

        if target_type is not None and not isinstance(target_type, Sequence):
            print("  [Fix 1] Normalizing reward_model.ground_truth.target: string -> Sequence(string)")

            def fix_target(example):
                t = example["reward_model"]["ground_truth"]["target"]
                if isinstance(t, str):
                    t = [t]
                return {
                    "reward_model": {
                        "ground_truth": {"target": t},
                        "style": example["reward_model"]["style"],
                    }
                }

            new_features = ds.features.copy()
            new_features["reward_model"] = {
                "ground_truth": {"target": Sequence(Value("string"))},
                "style": Value("string"),
            }
            ds = ds.map(fix_target, features=Features(new_features))
            changed = True
        else:
            print("  [Fix 1] reward_model.ground_truth.target already Sequence — skipping")

    # ------------------------------------------------------------------ #
    # Fix 2: extra_info — drop support_docs if present
    # ------------------------------------------------------------------ #
    ei_feature = ds.features.get("extra_info")
    if ei_feature is not None:
        has_support_docs = False
        if hasattr(ei_feature, "feature"):
            has_support_docs = "support_docs" in ei_feature.feature
        elif isinstance(ei_feature, dict):
            has_support_docs = "support_docs" in ei_feature

        if has_support_docs:
            print("  [Fix 2] Dropping extra_info.support_docs")

            def drop_support_docs(example):
                ei = dict(example["extra_info"])
                ei.pop("support_docs", None)
                return {"extra_info": ei}

            new_features = ds.features.copy()
            new_features["extra_info"] = {
                "index": Value("int64"),
                "split": Value("string"),
            }
            ds = ds.map(drop_support_docs, features=Features(new_features))
            changed = True
        else:
            print("  [Fix 2] extra_info has no support_docs — skipping")

    # ------------------------------------------------------------------ #
    # Fix 3: golden_answers — ensure Sequence(string)
    # ------------------------------------------------------------------ #
    ga_feature = ds.features.get("golden_answers")
    if ga_feature is not None and not isinstance(ga_feature, Sequence):
        print("  [Fix 3] Normalizing golden_answers: scalar -> Sequence(string)")

        def fix_golden_answers(example):
            ga = example["golden_answers"]
            if isinstance(ga, str):
                ga = [ga]
            return {"golden_answers": ga}

        new_features = ds.features.copy()
        new_features["golden_answers"] = Sequence(Value("string"))
        ds = ds.map(fix_golden_answers, features=Features(new_features))
        changed = True
    else:
        print("  [Fix 3] golden_answers already Sequence or absent — skipping")

    # ------------------------------------------------------------------ #
    # Write output
    # ------------------------------------------------------------------ #
    print(f"  Final features:\n  {ds.features}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_parquet(output_path)
    status = "FIXED" if changed else "unchanged"
    print(f"  [{status}] Written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Normalize parquet schema for HuggingFace concatenation")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing input .parquet files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write normalized .parquet files")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Specific filenames to process (default: all .parquet in data_dir)")
    args = parser.parse_args()

    if args.files:
        parquet_files = [os.path.join(args.data_dir, f) for f in args.files]
    else:
        parquet_files = sorted([
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir)
            if f.endswith(".parquet")
        ])

    if not parquet_files:
        print(f"No .parquet files found in {args.data_dir}")
        return

    print(f"Found {len(parquet_files)} parquet file(s) to process.")

    for input_path in parquet_files:
        fname = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, fname)
        normalize_parquet(input_path, output_path)

    print("\nDone. Update your train script:")
    print(f"  export TEST_DATA_DIR=\"{args.output_dir}\"")


if __name__ == "__main__":
    main()
