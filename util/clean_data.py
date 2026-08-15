"""Drop all-zero numeric columns from a combined datagen CSV.

Usage:
    python util/clean_data.py data.csv cleaned.csv
"""

import argparse

import polars as pl


def main():
    parser = argparse.ArgumentParser(
        description="Remove numeric columns whose every value is zero"
    )
    parser.add_argument("input_csv", help="CSV to clean")
    parser.add_argument("output_csv", help="where to write the cleaned CSV")
    args = parser.parse_args()

    df = pl.read_csv(args.input_csv)
    kept = [
        col
        for col in df.columns
        if not (df[col].dtype.is_numeric() and (df[col] == 0).all())
    ]
    dropped = [col for col in df.columns if col not in kept]
    if dropped:
        print(f"Dropping {len(dropped)} all-zero columns: {dropped}")

    df.select(kept).write_csv(args.output_csv)
    print(f"Wrote {args.output_csv}: {df.shape[0]} rows, {len(kept)} columns")


if __name__ == "__main__":
    main()
