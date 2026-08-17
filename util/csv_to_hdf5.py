"""Migration tool: convert a pre-HDF5 datagen CSV into the layout the ML side reads.

**The generators no longer produce CSV** — they write this layout directly (see
`data_generation/dataset_io.py`). This exists for datasets generated before that
change, including the BEAVRS runs, so they can be brought onto the same format
rather than stranded. Delete it once nothing on disk is still CSV.

The layout is matched by `nuclear_surrogates.datamodule.dataset_helper.read_h5_file`:

    numeric_data      (rows, n_numeric) float64, gzip-compressed
    numeric_columns   names of the numeric columns, in order
    string_<col>      one dataset per non-numeric column
    string_columns    names of the non-numeric columns
    all_columns       every column name, in the CSV's original order

Usage:
    uv run --extra ml python util/csv_to_hdf5.py data.csv output.h5
    uv run --extra ml python util/csv_to_hdf5.py data.csv output.h5 --inspect
"""

import argparse

import h5py
import numpy as np
import polars as pl


def convert(input_csv, output_h5):
    print(f"Reading {input_csv}...")
    df = pl.read_csv(input_csv)

    # run_label only identifies the generating worker; the ML side drops it and
    # detects run boundaries from time resets instead.
    if "run_label" in df.columns:
        df = df.drop("run_label")

    numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
    string_cols = [col for col in df.columns if not df[col].dtype.is_numeric()]
    string_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_h5, "w") as f:
        if numeric_cols:
            f.create_dataset(
                "numeric_data",
                data=df.select(numeric_cols).to_numpy().astype(np.float64),
                compression="gzip",
                compression_opts=9,
            )
            f.create_dataset("numeric_columns", data=numeric_cols, dtype=string_dt)

        if string_cols:
            for col in string_cols:
                values = [
                    str(val) if val is not None else "" for val in df[col].to_list()
                ]
                f.create_dataset(f"string_{col}", data=values, dtype=string_dt)
            f.create_dataset("string_columns", data=string_cols, dtype=string_dt)

        f.create_dataset("all_columns", data=df.columns, dtype=string_dt)

    print(
        f"Wrote {output_h5}: {df.shape[0]} rows, "
        f"{len(numeric_cols)} numeric + {len(string_cols)} string columns"
    )


def inspect(path):
    """Print a compact summary of the file just written."""
    with h5py.File(path, "r") as f:
        print(f"\nDatasets in {path}:")
        for key in f:
            print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")

        columns = [
            col.decode() if isinstance(col, bytes) else col
            for col in f["all_columns"][:]
        ]
        preview = ", ".join(columns[:10])
        suffix = ", ..." if len(columns) > 10 else ""
        print(f"  {len(columns)} columns: {preview}{suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a datagen CSV to the HDF5 layout the ML pipeline reads"
    )
    parser.add_argument("input_csv", help="a CSV from before the HDF5 switch")
    parser.add_argument("output_h5", help="where to write the HDF5 file")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="print a summary of the written file",
    )
    args = parser.parse_args()

    convert(args.input_csv, args.output_h5)
    if args.inspect:
        inspect(args.output_h5)


if __name__ == "__main__":
    main()
