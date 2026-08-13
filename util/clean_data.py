import polars as pl

# Read the CSV
df = pl.read_csv("data.csv")

# Remove columns where all values are zero
df_cleaned = df.select(
    [
        col
        for col in df.columns
        if not (df[col].dtype.is_numeric() and (df[col] == 0).all())
    ]
)

# Save the cleaned data
df_cleaned.write_csv("cleaned.csv")
