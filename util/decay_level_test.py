"""Sweep the decay-chain truncation level and record how many nuclides survive.

`Chain.reduce(keep, level)` keeps *keep* plus everything reachable within
*level* decay/reaction steps. This sweeps the level and plots the curve, which
is how the truncation level used for generation was chosen: the count climbs
steeply from the keep-list size and then flattens once nothing new is reachable.

    uv run --extra sim python util/decay_level_test.py \
        data/chain_endfb71_pwr.xml out/ --max-level 90

Writes `nuclide_counts_vs_level.csv` and `.png` into the output directory.
`reduce_decay_chains.py` produces the trimmed chain itself at one chosen level.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import openmc.deplete

# The 95-nuclide set the CASL generation was built around: the actinides that
# matter for the breeding chain, the main absorbers, and the fission products
# with appreciable thermal cross sections.
DEFAULT_KEEP = [
    "H1", "B10", "B11", "N14", "O16", "Kr83",
    "Zr91", "Nb93", "Zr93", "Zr94", "Mo95", "Zr95",
    "Nb95", "Zr96", "Mo97", "Mo98", "Tc99", "Mo99",
    "Mo100", "Ru101", "Ru102", "Rh103", "Ru103", "Ru104",
    "Rh105", "Pd105", "Ru106", "Pd107", "Pd108", "Ag109",
    "Cd113", "In115", "Sn126", "I127", "I129", "Xe131",
    "Cs133", "Xe133", "Cs134", "I135", "Xe135", "Cs135",
    "Cs137", "La139", "Ba140", "Ce141", "Pr141", "Ce142",
    "Pr143", "Nd143", "Ce143", "Ce144", "Nd144", "Nd145",
    "Nd146", "Nd147", "Pm147", "Sm147", "Pm148", "Nd148",
    "Pm149", "Sm149", "Sm150", "Sm151", "Eu151", "Sm152",
    "Gd152", "Eu153", "Sm153", "Eu154", "Gd154", "Eu155",
    "Gd155", "Gd156", "Eu156", "Gd157", "Gd158", "Gd160",
    "U234", "U235", "U236", "Np237", "U238", "Pu238",
    "Pu239", "Pu240", "Pu241", "Am241", "Pu242", "Am242",
    "Cm242", "Am242_m1", "Am243", "Cm243", "Cm244",
]  # fmt: skip


def sweep(chain, keep_nuclides, max_level):
    """Nuclide count after `reduce` at every level from 0 to *max_level*."""
    levels = list(range(max_level + 1))
    counts = []
    for level in levels:
        counts.append(len(chain.reduce(keep_nuclides, level).nuclides))
        print(f"{level / max_level * 100:.0f}% complete", end="\r")
    print()
    return levels, counts


def main():
    parser = argparse.ArgumentParser(
        description="Sweep chain truncation level and plot the surviving count"
    )
    parser.add_argument("chain_file", help="depletion chain XML to read")
    parser.add_argument("output_dir", help="where to write the CSV and plot")
    parser.add_argument(
        "--max-level", type=int, default=90, help="highest level to sweep (default: 90)"
    )
    args = parser.parse_args()

    chain = openmc.deplete.Chain.from_xml(args.chain_file)
    print(f"{args.chain_file}: {len(chain.nuclides)} nuclides before trimming")

    levels, counts = sweep(chain, DEFAULT_KEEP, args.max_level)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "nuclide_counts_vs_level.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "nuclides_after_trimming"])
        writer.writerows(zip(levels, counts, strict=True))
    print(f"Table saved to {csv_path}")

    plt.figure()
    plt.plot(levels, counts, marker="o")
    plt.xlabel("decay levels included")
    plt.ylabel("number of nuclides tracked after trimming")
    plt.title("Nuclides tracked vs. decay levels")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "nuclide_counts_vs_level.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
