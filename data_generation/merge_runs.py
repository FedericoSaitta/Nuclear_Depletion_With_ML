"""Merge the per-worker HDF5 files a generation sweep produced into one dataset.

Each worker writes its own file so the processes never contend for a handle;
this is the step that turns a directory of them into the single file training
reads.

    uv run --extra sim python data_generation/merge_runs.py \
        data_generation/data datasets/casl_runs.h5

    # or, dropping the nuclides the chain tracks but this run never produced:
    uv run --extra sim python data_generation/merge_runs.py \
        data_generation/data datasets/casl_runs.h5 --drop-empty

Needs no OpenMC, so it runs in either environment.
"""

import argparse
import glob
import os

from loguru import logger

import dataset_io


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-worker depletion HDF5 files into one dataset"
    )
    parser.add_argument("input_dir", help="directory of worker_*.h5 files")
    parser.add_argument("output_h5", help="where to write the merged dataset")
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="drop numeric columns that are zero in every row",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"No such directory: {args.input_dir}")

    # Sorted so a merge is reproducible: the row order of the output is the file
    # order, and glob's is filesystem-dependent.
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))
    if not paths:
        raise SystemExit(f"No .h5 files in {args.input_dir}")

    logger.info(f"Merging {len(paths)} files from {args.input_dir}")

    data = dataset_io.merge_frames(paths)
    if args.drop_empty:
        # After merging, not per file: a nuclide can be zero for one worker and
        # non-zero for another, and dropping per file would lose it.
        data = dataset_io.drop_empty_columns(data)
    dataset_io.write_run_h5(data, args.output_h5)


if __name__ == "__main__":
    main()
