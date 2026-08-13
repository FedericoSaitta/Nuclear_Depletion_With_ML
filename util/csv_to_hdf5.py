import polars as pl
import h5py
import numpy as np

# Read the CSV
print("Reading CSV...")
df = pl.read_csv("data.csv")

# Remove run_label column if it exists
if "run_label" in df.columns:
    print("Removing 'run_label' column...")
    df = df.drop("run_label")

print(f"Total columns: {len(df.columns)}")
print(f"Total rows: {len(df)}")

# Separate numeric and non-numeric columns
numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
string_cols = [col for col in df.columns if not df[col].dtype.is_numeric()]

print(f"\nNumeric columns: {len(numeric_cols)}")
print(f"Non-numeric columns: {len(string_cols)}")

# Create HDF5 file
print("\nWriting to HDF5...")
with h5py.File("output.h5", "w") as f:
    # Store numeric data
    if numeric_cols:
        numeric_data = df.select(numeric_cols).to_numpy().astype(np.float64)
        f.create_dataset(
            "numeric_data", data=numeric_data, compression="gzip", compression_opts=9
        )

        # Store numeric column names
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("numeric_columns", data=numeric_cols, dtype=dt)

    # Store string data (if any)
    if string_cols:
        for col in string_cols:
            # Convert to list of strings to handle encoding properly
            string_list = [
                str(val) if val is not None else "" for val in df[col].to_list()
            ]
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset(f"string_{col}", data=string_list, dtype=dt)

        # Store string column names
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("string_columns", data=string_cols, dtype=dt)

    # Store all column names in order
    dt = h5py.string_dtype(encoding="utf-8")
    f.create_dataset("all_columns", data=df.columns, dtype=dt)

print("Done!")

# Detailed inspection of the HDF5 file
print("\n" + "=" * 60)
print("DETAILED HDF5 FILE INSPECTION")
print("=" * 60)

with h5py.File("output.h5", "r") as f:
    # Show all datasets in the file
    print("\nDatasets in file:")
    for key in f.keys():
        dataset = f[key]
        if hasattr(dataset, "shape"):
            print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        else:
            print(f"  - {key}")

    # Get all column names
    all_columns = [col for col in f["all_columns"][:]]

    print(f"\nTotal number of columns: {len(all_columns)}")

    # Get numeric data
    if "numeric_data" in f:
        numeric_data = f["numeric_data"][:]
        numeric_columns = [col for col in f["numeric_columns"][:]]

        print(f"\nNumeric data shape: {numeric_data.shape}")
        print(f"Number of numeric columns: {len(numeric_columns)}")
        print(f"Total number of rows: {numeric_data.shape[0]}")

        print("\nNumeric column names:")
        for i, col in enumerate(numeric_columns):
            print(f"  {i}: {col}")

        print("\nFirst 5 rows of numeric data:")
        print(numeric_data[:5])

        print("\nSample values for each numeric column (first row):")
        for i, col in enumerate(numeric_columns[:10]):  # Show first 10
            print(f"  {col}: {numeric_data[0, i]}")

        print("\nBasic statistics for first 5 numeric columns:")
        for i in range(min(5, len(numeric_columns))):
            col_data = numeric_data[:, i]
            print(f"\n  Column: {numeric_columns[i]}")
            print(f"    Min: {np.min(col_data)}")
            print(f"    Max: {np.max(col_data)}")
            print(f"    Mean: {np.mean(col_data)}")
            print(f"    Std: {np.std(col_data)}")

    # Get string data
    if "string_columns" in f:
        string_columns = [col for col in f["string_columns"][:]]

        print(f"\n\nString columns ({len(string_columns)}):")
        for col in string_columns:
            string_data = f[f"string_{col}"][:]
            # Decode if necessary
            if string_data.dtype.kind == "S":
                string_data = [
                    s.decode("utf-8") if isinstance(s, bytes) else s
                    for s in string_data
                ]
            print(f"\n  Column: {col}")
            print(f"    First 5 values: {string_data[:5]}")
            print(f"    Unique values: {len(np.unique(string_data))}")
