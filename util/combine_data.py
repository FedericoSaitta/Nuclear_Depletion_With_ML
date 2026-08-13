# This Python script combines the data files produced by the data generation Python file.

import os
import glob
import pandas as pd


def merge_csv_files(input_dir, output_file):
    # Ensure the directory exists
    if not os.path.isdir(input_dir):
        raise ValueError(f"Directory does not exist: {input_dir}")

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        print("No CSV files found in the directory.")
        return

    # Sort the CSV files alphabetically
    csv_files.sort()

    # Read and concatenate all CSV files
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"Added: {file}")
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    merged_df = pd.concat(df_list, ignore_index=True)

    # Save to output file
    merged_df.to_csv(output_file, index=False)
    print(f"All CSV files merged into: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge all CSV files in a directory.")
    parser.add_argument("input_dir", help="Directory containing CSV files to merge")
    parser.add_argument("output_file", help="Path to the merged CSV output")
    args = parser.parse_args()

    merge_csv_files(args.input_dir, args.output_file)
    # Usage example:
    # python3 util/combine_data.py data_generation/data data.csv
