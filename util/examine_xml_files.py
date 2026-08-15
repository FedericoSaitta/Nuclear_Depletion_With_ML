import xml.etree.ElementTree as ET


def find_production_pathways(chain_file, target_nuclide, mode="production"):
    """
    Find all reactions and decays that produce or destroy a target nuclide.

    Parameters:
    -----------
    chain_file : str
        Path to the depletion chain XML file
    target_nuclide : str
        Target nuclide name (e.g., 'U238', 'Pu239')
    mode : str
        'production' - find pathways that CREATE the target nuclide
        'destruction' - find pathways that DESTROY/CONSUME the target nuclide

    Returns:
    --------
    dict : Dictionary with 'reactions' and 'decays' lists
    """
    tree = ET.parse(chain_file)
    root = tree.getroot()

    results = {"reactions": [], "decays": []}

    if mode == "production":
        # PRODUCTION MODE: Find where target_nuclide is the PRODUCT
        for nuclide in root.findall(".//nuclide"):
            parent_name = nuclide.get("name")
            half_life = nuclide.get("half_life", "stable")
            decay_constant = nuclide.get("decay_constant", "0")

            # Check reactions
            for reaction in nuclide.findall(".//reaction"):
                reaction_type = reaction.get("type")
                target_attr = reaction.get("target")
                Q_value = reaction.get("Q", "N/A")
                branching_ratio = reaction.get("branching_ratio", "1.0")

                if target_attr == target_nuclide:
                    results["reactions"].append(
                        {
                            "parent": parent_name,
                            "type": reaction_type,
                            "target": target_nuclide,
                            "Q_value": Q_value,
                            "branching_ratio": branching_ratio,
                        }
                    )

                # Check for target in child elements
                for product in reaction.findall(".//product"):
                    if product.text == target_nuclide:
                        results["reactions"].append(
                            {
                                "parent": parent_name,
                                "type": reaction_type,
                                "target": target_nuclide,
                                "Q_value": Q_value,
                                "branching_ratio": branching_ratio,
                            }
                        )

            # Check decay modes
            for decay in nuclide.findall(".//decay"):
                decay_type = decay.get("type")
                target_attr = decay.get("target")
                branching_ratio = decay.get("branching_ratio", "1.0")

                if target_attr == target_nuclide:
                    results["decays"].append(
                        {
                            "parent": parent_name,
                            "type": decay_type,
                            "target": target_nuclide,
                            "half_life": half_life,
                            "decay_constant": decay_constant,
                            "branching_ratio": branching_ratio,
                        }
                    )

                # Check for target in child elements
                for product in decay.findall(".//product"):
                    if product.text == target_nuclide:
                        results["decays"].append(
                            {
                                "parent": parent_name,
                                "type": decay_type,
                                "target": target_nuclide,
                                "half_life": half_life,
                                "decay_constant": decay_constant,
                                "branching_ratio": branching_ratio,
                            }
                        )

    elif mode == "destruction":
        # DESTRUCTION MODE: Find where target_nuclide is the PARENT/REACTANT
        for nuclide in root.findall(".//nuclide"):
            parent_name = nuclide.get("name")

            # Only process if this IS the target nuclide
            if parent_name == target_nuclide:
                half_life = nuclide.get("half_life", "stable")
                decay_constant = nuclide.get("decay_constant", "0")

                # All reactions from this nuclide destroy it
                for reaction in nuclide.findall(".//reaction"):
                    reaction_type = reaction.get("type")
                    target_attr = reaction.get("target")
                    Q_value = reaction.get("Q", "N/A")
                    branching_ratio = reaction.get("branching_ratio", "1.0")

                    # Get product from target attribute or child elements
                    product_nuclide = target_attr
                    if not product_nuclide:
                        product_elem = reaction.find(".//product")
                        if product_elem is not None:
                            product_nuclide = product_elem.text

                    results["reactions"].append(
                        {
                            "parent": parent_name,
                            "type": reaction_type,
                            "product": product_nuclide,
                            "Q_value": Q_value,
                            "branching_ratio": branching_ratio,
                        }
                    )

                # All decays from this nuclide destroy it
                for decay in nuclide.findall(".//decay"):
                    decay_type = decay.get("type")
                    target_attr = decay.get("target")
                    branching_ratio = decay.get("branching_ratio", "1.0")

                    # Get product from target attribute or child elements
                    product_nuclide = target_attr
                    if not product_nuclide:
                        product_elem = decay.find(".//product")
                        if product_elem is not None:
                            product_nuclide = product_elem.text

                    results["decays"].append(
                        {
                            "parent": parent_name,
                            "type": decay_type,
                            "product": product_nuclide,
                            "half_life": half_life,
                            "decay_constant": decay_constant,
                            "branching_ratio": branching_ratio,
                        }
                    )

    else:
        raise ValueError("mode must be either 'production' or 'destruction'")

    return results


def _report(chain_file, target, mode):
    """Print every pathway that produces or destroys *target*."""
    verb = "CREATE" if mode == "production" else "DESTROY"

    print("=" * 60)
    print(f"{mode.upper()} PATHWAYS FOR {target}")
    print("=" * 60)
    pathways = find_production_pathways(chain_file, target, mode=mode)

    if pathways["reactions"]:
        print(f"\nREACTIONS THAT {verb} {target}:")
        for rxn in pathways["reactions"]:
            product = rxn.get("product", rxn["target"])
            print(f"  {rxn['parent']} + n → {product} (via {rxn['type']})")
            print(
                f"    Q-value: {rxn['Q_value']} MeV, "
                f"Branching: {rxn['branching_ratio']}"
            )
    else:
        print("\nREACTIONS: None found")

    if pathways["decays"]:
        print(f"\nDECAYS THAT {verb} {target}:")
        for decay in pathways["decays"]:
            product = decay.get("product", decay["target"])
            print(f"  {decay['parent']} → {product} (via {decay['type']})")
            print(
                f"    Half-life: {decay['half_life']} s, "
                f"Branching: {decay['branching_ratio']}"
            )
    else:
        print("\nDECAYS: None found")

    total = len(pathways["reactions"]) + len(pathways["decays"])
    print(f"\nTotal {mode} pathways: {total}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="List the reactions and decays that produce or destroy a nuclide"
    )
    parser.add_argument(
        "chain_file", help="depletion chain XML, e.g. data/chain_casl_pwr.xml"
    )
    parser.add_argument("target", help="nuclide to trace, e.g. Pu242")
    args = parser.parse_args()

    _report(args.chain_file, args.target, "production")
    print()
    _report(args.chain_file, args.target, "destruction")


if __name__ == "__main__":
    main()
