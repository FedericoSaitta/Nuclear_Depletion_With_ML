"""The OpenMC pin-cell model both depletion pipelines build.

Four concentric regions — fuel, helium gap, Zircaloy cladding, borated water —
inside a reflective box, so the model is an infinite lattice of identical pins.

**One knob distinguishes the pipelines' models: `model.symmetry`.** `full` puts
the whole cell in the problem with boundaries at ±pitch/2; `quarter` cuts on the
two symmetry planes through the pin centre and models one quadrant, with
boundaries at 0 and +pitch/2. Both describe the same infinite lattice, and the
quarter costs a quarter of the tracking. Everything else — radii, densities,
thermal scattering, the source spectrum — is shared by construction.

Keeping every pipeline on one model is the point of the module. A surrogate
trained on one pipeline's data and evaluated against another's is only measuring
the physics if both describe the same fuel pin, and a geometry difference leaves
no trace in the generated columns for anything downstream to catch.

What does differ between the pipelines is the *history* they apply — sampled per
step versus measured and fixed — and that lives in the entry points, not here.
"""

import math

import openmc

from config import SYMMETRY_FRACTION
from nuclides import CAPTURE_NUCLIDES, FISSION_NUCLIDES

# The initial source guess, which has to sit inside the fuel. The origin is a
# reflective corner of the quarter cell rather than the pin centre, so that
# symmetry starts off-axis.
SOURCE_POINT = {"full": (0.0, 0.0, 0.0), "quarter": (0.05, 0.05, 0.0)}


def update_water_composition(water, boron_ppm, density_g_cm3):
    """Set the moderator's soluble-boron content and density.

    Boron is carried as a weight fraction of the water, so H and O are scaled
    down to make room for it. At 0 ppm this reproduces pure H2O exactly, which is
    what the measured-history pipelines run with.
    """
    # Base water weight fractions (pure H2O)
    h_weight_frac = 0.111894  # 2*1.008 / (2*1.008 + 15.999)
    o_weight_frac = 0.888106  # 15.999 / (2*1.008 + 15.999)
    boron_weight_frac = boron_ppm * 1e-6

    # Remove all existing elements
    water.remove_element("H")
    water.remove_element("O")

    # Check if boron exists before trying to remove
    if any(elem[0] == "B" for elem in water.get_elements()):
        water.remove_element("B")

    # Add elements with proper normalization
    if boron_weight_frac > 0:
        # Scale H and O down to make room for boron
        water.add_element("H", h_weight_frac * (1 - boron_weight_frac), "wo")
        water.add_element("O", o_weight_frac * (1 - boron_weight_frac), "wo")
        water.add_element("B", boron_weight_frac, "wo")
    else:
        # Pure water, no boron
        water.add_element("H", h_weight_frac, "wo")
        water.add_element("O", o_weight_frac, "wo")

    # Set density
    water.set_density("g/cm3", density_g_cm3)


def set_temperatures(materials, fuel_temp, clad_temp, mod_temp):
    """Apply the thermal state to the four materials.

    The gap is a hundredth of a millimetre of helium pressed against the pellet,
    so it is taken at the fuel temperature.
    """
    fuel, gap, clad, water = materials
    fuel.temperature = fuel_temp
    gap.temperature = fuel_temp
    clad.temperature = clad_temp
    water.temperature = mod_temp


def create_materials(cfg):
    """Fuel, helium gap, Zircaloy cladding and borated water.

    Temperatures, moderator density and boron are *state*, not model. A config
    that fixes them for the whole run — the BEAVRS ones — has them applied here;
    a pipeline that redraws them every step (`datagen.py`) leaves them out of the
    config and re-applies them with `set_temperatures` and
    `update_water_composition` before each transport solve. The fallbacks below
    are therefore only ever what a sampled run holds before its first step.
    """
    fuel = openmc.Material(name="uo2")
    fuel.add_element("U", 1, percent_type="ao", enrichment=cfg["enrichment"])
    fuel.add_element("O", 2)
    fuel.set_density("g/cc", cfg["fuel_density"])
    fuel.depletable = True

    gap = openmc.Material(name="gap")
    gap.add_element("He", 1.0)
    gap.set_density("g/cc", 0.000178)  # helium at room temperature and pressure

    clad = openmc.Material(name="clad")
    clad.add_element("Zr", 1)
    clad.set_density("g/cc", 6.56)  # Zircaloy-4

    water = openmc.Material(name="water")
    # A placeholder element first, so update_water_composition has an H to
    # remove; it writes the real composition immediately below.
    water.add_element("H", 1.0, "wo")
    update_water_composition(water, 0.0, cfg.get("mod_density", 1.0))
    water.add_s_alpha_beta("c_H_in_H2O")  # thermal scattering for bound hydrogen

    materials = (fuel, gap, clad, water)
    set_temperatures(
        materials, cfg.get("fuel_temp"), cfg.get("clad_temp"), cfg.get("mod_temp")
    )
    return materials


# Note these are called volumes but as this is a 2D problem they are effectively
# areas.
def set_volumes(materials, radii, pitch, symmetry):
    """Set material volumes for *symmetry*.

    radii = [fuel outer, gap outer, clad outer].

    Every region scales by the same fraction, the water box included:
    (pitch/2)**2 is exactly a quarter of pitch**2.
    """
    fuel, gap, clad, water = materials
    fraction = SYMMETRY_FRACTION[symmetry]

    fuel.volume = fraction * math.pi * radii[0] ** 2
    gap.volume = fraction * math.pi * (radii[1] ** 2 - radii[0] ** 2)
    clad.volume = fraction * math.pi * (radii[2] ** 2 - radii[1] ** 2)
    water.volume = fraction * (pitch**2 - math.pi * radii[2] ** 2)


def create_geometry(materials, radii, pitch, symmetry):
    """Concentric fuel | gap | cladding | water inside a reflective box.

    radii = [fuel outer, gap outer, clad outer]. All four boundaries are
    reflective in both symmetries; only where the two lower ones sit differs.
    """
    fuel, gap, clad, water = materials

    fuel_or = openmc.ZCylinder(r=radii[0], name="fuel_outer")
    gap_or = openmc.ZCylinder(r=radii[1], name="gap_outer")
    clad_or = openmc.ZCylinder(r=radii[2], name="clad_outer")

    half_pitch = pitch / 2.0
    # 0 for the quarter cell, which is cut on the planes through the pin centre;
    # -pitch/2 for the full one, which is centred on the pin. Indexed rather
    # than compared so an unvalidated symmetry raises instead of silently
    # falling through to a full cell.
    lower = {"full": -half_pitch, "quarter": 0.0}[symmetry]
    x_min = openmc.XPlane(x0=lower, boundary_type="reflective", name="x_min")
    x_max = openmc.XPlane(x0=half_pitch, boundary_type="reflective", name="x_max")
    y_min = openmc.YPlane(y0=lower, boundary_type="reflective", name="y_min")
    y_max = openmc.YPlane(y0=half_pitch, boundary_type="reflective", name="y_max")

    box = +x_min & -x_max & +y_min & -y_max

    cells = [
        openmc.Cell(name="fuel", fill=fuel, region=-fuel_or & box),
        openmc.Cell(name="gap", fill=gap, region=+fuel_or & -gap_or & box),
        openmc.Cell(name="clad", fill=clad, region=+gap_or & -clad_or & box),
        openmc.Cell(name="water", fill=water, region=+clad_or & box),
    ]
    return openmc.Geometry(openmc.Universe(cells=cells))


def create_settings(cfg, symmetry, tally_output=False):
    """OpenMC settings for one transport solve.

    `verbosity` and `output` are set here rather than patched onto the returned
    object by the caller. `tally_output` writes the human-readable `tallies.out`
    and stays off unless the pipeline actually scores tallies.
    """
    settings = openmc.Settings()
    settings.particles = cfg["particles"]
    settings.inactive = cfg["inactive"]
    settings.batches = cfg["batches"]
    settings.verbosity = 1
    settings.output = {"tallies": tally_output}

    source = openmc.IndependentSource()
    source.space = openmc.stats.Point(SOURCE_POINT[symmetry])
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Watt()
    settings.source = source

    settings.temperature = {"method": cfg.get("temp_method", "interpolation")}

    if cfg.get("seed") is not None:
        settings.seed = cfg["seed"]

    return settings


def create_tallies(fuel):
    """Create tallies for flux and reaction rates in the fuel.

    Uses high tally IDs (9001+) to avoid conflicts with the depletion
    operator's internal tallies.

    Returns an openmc.Tallies object with:
      - 'fuel_flux':     total neutron flux in fuel
      - 'fission_rates': fission rate per nuclide (FISSION_NUCLIDES)
      - 'capture_rates': (n,gamma) rate per nuclide (CAPTURE_NUCLIDES)

    CAPTURE_NUCLIDES covers every isotope in the 7-isotope chain
    (U238, U239, Np239, Pu239-Pu242) so the uncertainty analysis can
    build the Bateman matrix entirely from measured one-group rates.
    """
    tallies = openmc.Tallies()
    mat_filter = openmc.MaterialFilter(fuel)

    t_flux = openmc.Tally(tally_id=9001, name="fuel_flux")
    t_flux.filters = [mat_filter]
    t_flux.scores = ["flux"]
    tallies.append(t_flux)

    t_fission = openmc.Tally(tally_id=9002, name="fission_rates")
    t_fission.filters = [mat_filter]
    t_fission.nuclides = FISSION_NUCLIDES
    t_fission.scores = ["fission"]
    tallies.append(t_fission)

    t_capture = openmc.Tally(tally_id=9003, name="capture_rates")
    t_capture.filters = [mat_filter]
    t_capture.nuclides = CAPTURE_NUCLIDES
    t_capture.scores = ["(n,gamma)"]
    tallies.append(t_capture)

    return tallies


def build_model(cfg, with_tallies=False):
    """Everything a pipeline needs to run this config's pin cell.

    Returns `(materials_tuple, materials, geometry, settings, tallies)`. The four
    materials come back individually as well as inside the `openmc.Materials`
    collection, because the pipelines address them by name to apply each step's
    operating state. `tallies` is None unless asked for.
    """
    symmetry = cfg["symmetry"]
    radii, pitch = cfg["geometry_radii"], cfg["geometry_pitch"]

    materials_tuple = create_materials(cfg)
    set_volumes(materials_tuple, radii, pitch, symmetry)

    materials = openmc.Materials(list(materials_tuple))
    geometry = create_geometry(materials_tuple, radii, pitch, symmetry)
    settings = create_settings(cfg, symmetry, tally_output=with_tallies)
    tallies = create_tallies(materials_tuple[0]) if with_tallies else None

    return materials_tuple, materials, geometry, settings, tallies


def build_and_export_model(cfg, results_dir, with_tallies=False):
    """`build_model`, with the XML written into the worker's directory.

    The sampled-history pipeline rewrites materials.xml before every step —
    that is where its operating state lives — so what is exported here is only
    that run's starting point. Geometry and settings are final either way.
    """
    parts = build_model(cfg, with_tallies=with_tallies)
    _mats, materials, geometry, settings, _tallies = parts

    geometry.export_to_xml(path=results_dir)
    settings.export_to_xml(path=results_dir)
    materials.export_to_xml(path=results_dir)

    return parts
