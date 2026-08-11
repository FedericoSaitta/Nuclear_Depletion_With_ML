import openmc.deplete
import os

# Load the decay chain
script_dir = os.path.dirname(os.path.abspath(__file__))
chain_file = "../data/chain_casl_pwr.xml"
chain = openmc.deplete.Chain.from_xml(os.path.join(script_dir, chain_file))

# List all isotopes tracked
print(f"Number of nuclides tracked: {len(chain.nuclides)}\n")

print(f"Attributes of chain: {dir(chain)}")
print(f"Attributes of nuc: {dir(chain.nuclides[0])}")

# Iterate over all nuclides
for nuc in chain.nuclides:
    print(f"Nuclide: {nuc.name}, Half-life: {nuc.half_life}")

    # Iterate over decay modes
    for decay in nuc.decay_modes:
        decay_type, daughter, branching_ratio = decay
        print(f"  Decays to {daughter} via {decay_type}, BR={branching_ratio}")

    # Check fission yields
    if nuc.yield_data is not None:
        print(
            f"  Fission yield data available for {len(nuc.yield_data)} reactions/energies"
        )
