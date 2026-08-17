"""Drop all-zero numeric columns from a generated depletion dataset.

The depletion chain tracks a couple of hundred nuclides and a pin-cell run
produces exactly zero of most of them, so this usually removes most of the file.

    uv run --extra ml python util/clean_data.py data.h5 cleaned.h5

`data_generation/merge_runs.py --drop-empty` does the same thing as part of the
merge, which is where it usually belongs; this exists for datasets that are
already merged.
"""

import argparse
import os
import sys

# `data_generation` is not a package — its modules are imported by adding the
# directory to the path, the same way its own entry points do.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_generation"),
)

import dataset_io  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Remove numeric columns whose every value is zero"
    )
    parser.add_argument("input_h5", help="dataset to clean")
    parser.add_argument("output_h5", help="where to write the cleaned dataset")
    args = parser.parse_args()

    data = dataset_io.read_run_h5(args.input_h5)
    before = len(data)
    data = dataset_io.drop_empty_columns(data)
    dataset_io.write_run_h5(data, args.output_h5)
    print(f"{before} columns -> {len(data)}")


if __name__ == "__main__":
    main()
