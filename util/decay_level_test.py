# file: trim_chain_loop.py  (or paste into a notebook cell)

import os
import openmc.deplete
import matplotlib.pyplot as plt
import csv

# Load the decay chain
script_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
chain_file = "../data/chain_endfb71_pwr.xml"  # chain_endfb71_pwr # chain_casl_pwr
chain_path = os.path.join(script_dir, chain_file)
chain = openmc.deplete.Chain.from_xml(chain_path)

# --- isotope keep-list (constant across all levels) ---

keep_nuclides = [
    "H1",
    "B10",
    "B11",
    "N14",
    "O16",
    "Kr83",
    "Zr91",
    "Nb93",
    "Zr93",
    "Zr94",
    "Mo95",
    "Zr95",
    "Nb95",
    "Zr96",
    "Mo97",
    "Mo98",
    "Tc99",
    "Mo99",
    "Mo100",
    "Ru101",
    "Ru102",
    "Rh103",
    "Ru103",
    "Ru104",
    "Rh105",
    "Pd105",
    "Ru106",
    "Pd107",
    "Pd108",
    "Ag109",
    "Cd113",
    "In115",
    "Sn126",
    "I127",
    "I129",
    "Xe131",
    "Cs133",
    "Xe133",
    "Cs134",
    "I135",
    "Xe135",
    "Cs135",
    "Cs137",
    "La139",
    "Ba140",
    "Ce141",
    "Pr141",
    "Ce142",
    "Pr143",
    "Nd143",
    "Ce143",
    "Ce144",
    "Nd144",
    "Nd145",
    "Nd146",
    "Nd147",
    "Pm147",
    "Sm147",
    "Pm148",
    "Nd148",
    "Pm149",
    "Sm149",
    "Sm150",
    "Sm151",
    "Eu151",
    "Sm152",
    "Gd152",
    "Eu153",
    "Sm153",
    "Eu154",
    "Gd154",
    "Eu155",
    "Gd155",
    "Gd156",
    "Eu156",
    "Gd157",
    "Gd158",
    "Gd160",
    "U234",
    "U235",
    "U236",
    "Np237",
    "U238",
    "Pu238",
    "Pu239",
    "Pu240",
    "Pu241",
    "Am241",
    "Pu242",
    "Am242",
    "Cm242",
    "Am242_m1",
    "Am243",
    "Cm243",
    "Cm244",
]


# --- results dir and output XML (overwritten each iteration) ---
results_dir = os.path.join(script_dir, "util_results", "trimmed_chain")
os.makedirs(results_dir, exist_ok=True)
output_chain_file = os.path.join(results_dir, "iterated_trimmed_chain.xml")

# --- sweep levels 0..90, collect counts ---
max_level = 90
level_numbers = list(range(max_level + 1))
nuclide_counts = []

for level in level_numbers:
    trimmed_chain = chain.reduce(keep_nuclides, level)
    # overwrite the same XML to avoid file spam
    trimmed_chain.export_to_xml(output_chain_file)
    count_after = len(trimmed_chain.nuclides)
    nuclide_counts.append(count_after)
    progress = level / max_level * 100
    print(f"{progress:.0f}% complete", end="\r")

# --- save table of results ---
csv_path = os.path.join(results_dir, "nuclide_counts_vs_level.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["level", "nuclides_after_trimming"])
    writer.writerows(zip(level_numbers, nuclide_counts))
print(f"Results table saved to: {csv_path}")

# --- plot level vs. number of nuclides ---
plt.figure()
plt.plot(level_numbers, nuclide_counts, marker="o")
plt.xlabel("decay levels included")
plt.ylabel("number of nuclides tracked after trimming")
plt.title("Nuclides Tracked vs. Decay Levels")
plt.grid(True)
plt.tight_layout()
plt.savefig()
