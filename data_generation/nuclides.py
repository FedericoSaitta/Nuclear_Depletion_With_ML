"""The nuclide lists the BEAVRS tallies are scored over.

Kept apart from `quarter_sim`, which builds the OpenMC objects, so that the
column schema can be imported — and tested — without OpenMC installed. That is
the whole reason this file exists: CI has no OpenMC build and no nuclear data,
so anything that needs them is untestable there.

Extending either list extends the generated dataset's columns automatically:
`tally_io.step_keys` derives the column names from them.
"""

# Energy released per fission [MeV], used to weight each nuclide's share of the
# fission power. Only nuclides that fission appreciably in a PWR thermal
# spectrum appear here.
FISSION_Q_VALUES = {
    "U235": 193.7,
    "U238": 198.5,
    "Pu239": 200.1,
    "Pu240": 196.9,
    "Pu241": 202.2,
}

FISSION_NUCLIDES = list(FISSION_Q_VALUES.keys())

# Capture tallies for every isotope in the 7-isotope breeding chain:
#
#     U238 --(n,γ)--> U239 --(β⁻)--> Np239 --(β⁻)--> Pu239
#                                                      |(n,γ)
#                                                      v
#                                Pu242 <--(n,γ)-- Pu241 <--(n,γ)-- Pu240
#                                                                    ^
#                                                                    |(n,γ)
#                                                                  Pu239
#
# Scoring all seven lets the Bateman matrix be built from measured one-group
# data rather than from literature α = σ_c/σ_f ratios for the Pu isotopes.
#
# U239 and Np239 are short-lived (minutes/days) but are valid (n,γ) tally
# targets provided the cross-section library covers them — ENDF/B-VIII.0 does.
# A library that lacks one will make OpenMC fail at initialisation; drop the
# offending nuclide from this list if that happens.
CAPTURE_NUCLIDES = ["U238", "U239", "Np239", "Pu239", "Pu240", "Pu241", "Pu242"]

# Fission is deliberately NOT scored for U239, Np239 and Pu242: their fission
# rates are negligible at thermal energies, so tallying them would add
# statistical noise without adding information to the matrix.
