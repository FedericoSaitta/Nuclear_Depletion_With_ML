import openmc.deplete
import os

# Load the decay chain
script_dir = os.path.dirname(os.path.abspath(__file__))
chain_file = "../data/chain_endfb71_pwr.xml"  # chain_endfb71_pwr # chain_casl_pwr
chain = openmc.deplete.Chain.from_xml(os.path.join(script_dir, chain_file))

# List all isotopes tracked
print(f"\nNumber of nuclides tracked before trimming: {len(chain.nuclides)}")


# note the levels range from 0-25. after which they die off.
levels = 25
print(f"Levels of decay tracked {levels}")

# get a chain just containing U235, U238, and O16 only
trimmed_chain = chain.reduce(
    [
        "U235",
        "U238",
        "O16",
    ],
    0,
)

# for the big table
# trimmed_chain = chain.reduce( ['H1', 'B10', 'B11', 'N14', 'O16', 'Kr83', 'Zr91','Nb93', 'Zr93', 'Zr94', 'Mo95', 'Zr95', 'Nb95','Zr96', 'Mo97', 'Mo98', 'Tc99', 'Mo99', 'Mo100','Ru101', 'Ru102', 'Rh103', 'Ru103', 'Ru104', 'Rh105','Pd105', 'Ru106', 'Pd107', 'Pd108', 'Ag109', 'Cd113','In115', 'Sn126', 'I127', 'I129', 'Xe131', 'Cs133','Xe133', 'Cs134', 'I135', 'Xe135', 'Cs135', 'Cs137','La139', 'Ba140', 'Ce141', 'Pr141', 'Ce142', 'Pr143','Nd143', 'Ce143', 'Ce144', 'Nd144', 'Nd145', 'Nd146','Nd147', 'Pm147', 'Sm147', 'Pm148', 'Nd148', 'Pm149','Sm149', 'Sm150', 'Sm151', 'Eu151', 'Sm152', 'Gd152','Eu153', 'Sm153', 'Eu154', 'Gd154', 'Eu155', 'Gd155','Gd156', 'Eu156', 'Gd157', 'Gd158', 'Gd160', 'U234','U235', 'U236', 'Np237', 'U238', 'Pu238', 'Pu239','Pu240', 'Pu241', 'Am241', 'Pu242', 'Am242', 'Cm242','Am242_m1', 'Am243', 'Cm243','Cm244'], levels)

# for the small table
# trimmed_chain = chain.reduce( ['B10', 'B11', 'O16', 'Kr83', 'Zr91', 'Zr93', 'Zr94', 'Mo95', 'Zr95', 'Nb95','Zr96', 'Mo97', 'Mo98', 'Tc99', 'Mo99', 'Mo100','Ru101', 'Ru102', 'Rh103', 'Ru103', 'Ru104', 'Rh105','Pd105', 'Ru106', 'Pd107', 'Pd108', 'Ag109', 'Cd113','In115', 'I127', 'I129', 'Xe131', 'Cs133','Xe133', 'Cs134', 'I135', 'Xe135', 'Cs135', 'Cs137','La139', 'Ba140', 'Ce141', 'Pr141', 'Ce142', 'Pr143','Nd143', 'Ce143', 'Ce144', 'Nd144', 'Nd145', 'Nd146','Nd147', 'Pm147', 'Sm147', 'Pm148', 'Nd148', 'Pm149','Sm149', 'Sm150', 'Sm151', 'Eu151', 'Sm152', 'Gd152','Eu153', 'Sm153', 'Eu154', 'Gd154', 'Eu155', 'Gd155','Gd156', 'Eu156', 'Gd157', 'Gd158', 'Gd160', 'U234','U235', 'U236', 'Np237', 'U238', 'Pu238', 'Pu239','Pu240', 'Pu241', 'Am241', 'Pu242', 'Am242', 'Cm242','Am242_m1', 'Am243', 'Cm243','Cm244'], levels)

# prepare the results directory
results_dir = os.path.join(script_dir, "util_results", "trimmed_chain")
os.makedirs(results_dir, exist_ok=True)

# export the trimmed chain to the just prepped util_results
output_chain_file = os.path.join(results_dir, "trimmed_chain.xml")
trimmed_chain.export_to_xml(output_chain_file)

# List all isotopes kept
new_chain = openmc.deplete.Chain.from_xml(output_chain_file)
print(f"Number of nuclides tracked after trimming: {len(new_chain.nuclides)}\n")
print(f"Trimmed chain saved to: {output_chain_file}")
