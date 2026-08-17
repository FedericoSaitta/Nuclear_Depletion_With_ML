"""Trim a depletion chain to a set of nuclides and their decay neighbourhood.

`Chain.reduce(keep, level)` keeps *keep* plus everything reachable from it
within *level* decay/reaction steps. Level 0 is the nuclides themselves; the
count climbs steeply and then flattens out around level 25, past which nothing
new is reachable — `decay_level_test.py` sweeps that curve.

    uv run --extra sim python util/reduce_decay_chains.py \
        data/chain_endfb71_pwr.xml out/trimmed.xml U235 U238 O16 --level 0

A smaller chain means a smaller depletion matrix and a faster run, at the cost
of not tracking what was cut.
"""

import argparse

import openmc.deplete


def main():
    parser = argparse.ArgumentParser(
        description="Trim a depletion chain to a nuclide set and its neighbourhood"
    )
    parser.add_argument("chain_file", help="depletion chain XML to read")
    parser.add_argument("output_file", help="where to write the trimmed chain")
    parser.add_argument(
        "nuclides", nargs="+", help="nuclides to keep, e.g. U235 U238 O16"
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="decay levels to follow out from the kept nuclides (default: 0)",
    )
    args = parser.parse_args()

    chain = openmc.deplete.Chain.from_xml(args.chain_file)
    print(f"{args.chain_file}: {len(chain.nuclides)} nuclides before trimming")
    print(f"Keeping {len(args.nuclides)} nuclides at level {args.level}")

    trimmed = chain.reduce(args.nuclides, args.level)
    trimmed.export_to_xml(args.output_file)

    # Read it back rather than trusting the in-memory object: this is the file
    # a simulation will actually load.
    written = openmc.deplete.Chain.from_xml(args.output_file)
    print(f"{args.output_file}: {len(written.nuclides)} nuclides after trimming")


if __name__ == "__main__":
    main()
