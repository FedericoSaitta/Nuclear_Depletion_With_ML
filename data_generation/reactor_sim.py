# This script sets up the openMC model used for data generation
import openmc
import math


def update_water_composition(water, boron_ppm, density_g_cm3):
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


def create_materials(config_dict, initial_boron_ppm=0, initial_water_density=1.0):
    enrichment = config_dict["enrichment"]
    fuel_density = config_dict["fuel_density"]  # in g/cm³

    # Create fuel material
    fuel = openmc.Material(name="uo2")
    fuel.add_element("U", 1, percent_type="ao", enrichment=enrichment)
    fuel.add_element("O", 2)
    fuel.set_density("g/cc", fuel_density)
    fuel.depletable = True

    # Create cladding material
    clad = openmc.Material(name="clad")
    clad.add_element("Zr", 1)
    clad.set_density("g/cc", 6)

    # Create water material
    water = openmc.Material(name="water")

    # (add a dummy element first so update_water_composition can remove it)
    water.add_element("H", 1.0, "wo")  # Temporary, will be replaced
    update_water_composition(water, initial_boron_ppm, initial_water_density)

    water.add_s_alpha_beta("c_H_in_H2O")  # Add thermal scattering for water

    return fuel, clad, water


# Note these are called volumes but as this is a 2D problem they are effectively areas
def set_material_volumes(fuel, clad, water, radii, pitch):
    fuel.volume = math.pi * radii[0] ** 2
    clad.volume = math.pi * (radii[1] ** 2 - radii[0] ** 2)
    water.volume = pitch**2 - math.pi * radii[1] ** 2


def create_geometry(materials, radii, pitch):
    pin_surfaces = [openmc.ZCylinder(r=r) for r in radii]
    pin_univ = openmc.model.pin(pin_surfaces, materials)

    half_width = pitch / 2
    left = openmc.XPlane(x0=-half_width, boundary_type="reflective")
    right = openmc.XPlane(x0=half_width, boundary_type="reflective")
    bottom = openmc.YPlane(y0=-half_width, boundary_type="reflective")
    top = openmc.YPlane(y0=half_width, boundary_type="reflective")

    bound_box = +left & -right & +bottom & -top
    root_cell = openmc.Cell(fill=pin_univ, region=bound_box)
    root_univ = openmc.Universe(cells=[root_cell])
    geometry = openmc.Geometry(root_univ)

    return geometry


def create_settings(config_dict):
    seed = config_dict["seed"]
    particles = config_dict["particles"]

    inactive = config_dict["inactive"]
    batches = config_dict["batches"]
    temp_method = config_dict["temp_method"]

    settings = openmc.Settings()
    settings.particles = particles
    settings.inactive = inactive
    settings.batches = batches

    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0, 0, 0))
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Watt()
    settings.source = source

    settings.temperature = {"method": temp_method}

    if seed is not None:
        settings.seed = seed

    return settings
