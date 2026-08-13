import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re


def load_and_plot_csvs(
    directory_path, x_column, y_columns, title="Multi-CSV Plot", output_dir="plots"
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all CSV files in the directory
    csv_files = list(Path(directory_path).glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return

    print(f"Found {len(csv_files)} CSV file(s)")

    # Ensure y_columns is a list
    if isinstance(y_columns, str):
        y_columns = [y_columns]

    # Create a color palette
    colors = plt.cm.tab10(range(len(csv_files)))

    # Store dataframes if needed for further analysis
    dataframes = {}

    # Read each CSV file
    for idx, csv_file in enumerate(csv_files):
        try:
            # Read CSV into pandas DataFrame
            df = pd.read_csv(csv_file)
            dataframes[csv_file.name] = df

            print(f"\nProcessing: {csv_file.name}")
            print(f"  Shape: {df.shape}")

            # Check if specified columns exist
            if x_column not in df.columns:
                print(f"  Warning: Column '{x_column}' not found in {csv_file.name}")
            for y_column in y_columns:
                if y_column not in df.columns:
                    print(
                        f"  Warning: Column '{y_column}' not found in {csv_file.name}"
                    )

        except Exception as e:
            print(f"  Error processing {csv_file.name}: {str(e)}")
            continue

    # Helper function to clean filename for legend
    def clean_filename(filename):
        """Remove _seed_ and everything after it from the filename"""
        return re.sub(r"_seed_.*$", "", filename)

    # Create plots for each y column
    for y_column in y_columns:
        # Create two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # --- PLOT 1: Normal Plot ---
        plotted_normal = 0
        for idx, csv_file in enumerate(csv_files):
            if csv_file.name not in dataframes:
                continue

            df = dataframes[csv_file.name]

            # Check if specified columns exist
            if x_column not in df.columns or y_column not in df.columns:
                continue

            # Clean the filename for the legend
            legend_label = clean_filename(csv_file.stem)

            # Plot the data
            ax1.plot(
                df[x_column],
                df[y_column],
                color=colors[idx],
                label=legend_label,
                marker="o",
                markersize=4,
                linewidth=2,
                alpha=0.7,
            )

            plotted_normal += 1

        # Customize plot 1
        ax1.set_xlabel(x_column, fontsize=12)
        ax1.set_ylabel(y_column, fontsize=12)
        ax1.set_title(f"{title} - {y_column}", fontsize=14, fontweight="bold")
        ax1.legend(loc="best", framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        print(f"\nPlot 1 ({y_column}): Normal plot with {plotted_normal} dataset(s)")

        # --- PLOT 2: Difference from Mean ---
        # Collect all y values to calculate mean across all datasets
        all_data = []
        for filename, df in dataframes.items():
            if x_column in df.columns and y_column in df.columns:
                all_data.append(df[[x_column, y_column]].copy())

        if len(all_data) > 0:
            # Calculate the mean across all datasets
            combined = pd.concat(all_data).groupby(x_column)[y_column].mean()
            print(f"Calculated mean across {len(all_data)} dataset(s)")

            plotted_diff = 0
            for idx, csv_file in enumerate(csv_files):
                if csv_file.name not in dataframes:
                    continue

                df = dataframes[csv_file.name]

                if x_column not in df.columns or y_column not in df.columns:
                    continue

                # Clean the filename for the legend
                legend_label = clean_filename(csv_file.stem)

                # Calculate difference from mean for this dataset
                df_aligned = df.set_index(x_column)
                differences = df_aligned[y_column] - combined

                # Plot the differences
                ax2.plot(
                    differences.index,
                    differences.values,
                    color=colors[idx],
                    label=legend_label,
                    marker="o",
                    markersize=4,
                    linewidth=0,
                    alpha=0.8,
                )

                plotted_diff += 1

            # Add zero reference line
            ax2.axhline(
                y=0,
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Mean",
                zorder=1,
            )

            # Styling for plot 2
            ax2.set_xlabel(x_column, fontsize=12)
            ax2.set_ylabel(f"{y_column} (Difference from Mean)", fontsize=12)
            ax2.set_title(
                f"{title} - {y_column} - Difference from Mean",
                fontsize=14,
                fontweight="bold",
            )
            ax2.legend(loc="best", framealpha=0.9)
            ax2.grid(True, alpha=0.3, linestyle="--")

            # Add light background shading for positive/negative regions
            ylim = ax2.get_ylim()
            ax2.set_ylim(ylim)

            print(
                f"Plot 2 ({y_column}): Difference from mean plot with {plotted_diff} dataset(s)"
            )
        else:
            ax2.text(
                0.5,
                0.5,
                "No valid data for difference plot",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=14,
            )
            ax2.set_title(
                f"{title} - {y_column} - Difference from Mean",
                fontsize=14,
                fontweight="bold",
            )

        plt.tight_layout()

        # Save the figure
        output_filename = f"{y_column}_comparison.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

        # Close the figure to free memory
        plt.close(fig)

    return dataframes


# Example usage
if __name__ == "__main__":
    # Specify your directory path
    directory = "data_generation/data"

    # Specify the columns you want to plot
    x_col = "time_days"  # Replace with your x-axis column name
    y_cols = [
        "k_eff",
        "power_W_g",
        "boron_ppm",
        "U233",
        "U235",
        "U238",
        "Xe135",
        "Pu241",
        "Pu239",
        "Sm149",
        "Th232",
    ]  # List of y-axis columns to plot

    # Load and plot all CSV files
    dfs = load_and_plot_csvs(
        directory_path=directory,
        x_column=x_col,
        y_columns=y_cols,
        title="Comparison of Multiple CSV Datasets",
    )

    # Optional: Access individual dataframes for further analysis
    if dfs:
        print("\n" + "=" * 50)
        print("Loaded dataframes:")
        for filename, df in dfs.items():
            print(f"  {filename}: {df.shape[0]} rows × {df.shape[1]} columns")
