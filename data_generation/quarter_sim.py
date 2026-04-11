"""
Reactor model setup for BEAVRS-based depletion data generation.

Quarter-pin geometry with tallies for flux and reaction rates.
No boron — water is pure H2O with S(α,β) thermal scattering.
"""
import openmc
import math


# Energy per fission [MeV] for power fraction calculation
FISSION_Q_VALUES = {
    'U235':  193.7,
    'U238':  198.5,
    'Pu239': 200.1,
    'Pu240': 196.9,
    'Pu241': 202.2,
}

FISSION_NUCLIDES = list(FISSION_Q_VALUES.keys())
CAPTURE_NUCLIDES = ['U238']


def create_materials(config):
    """Create fuel, gap, cladding, and water materials.

    Water is pure H2O (no boron).
    """
    enrichment   = config['enrichment']
    fuel_density = config['fuel_density']

    fuel = openmc.Material(name="uo2")
    fuel.add_element("U", 1, percent_type="ao", enrichment=enrichment)
    fuel.add_element("O", 2)
    fuel.set_density("g/cc", fuel_density)
    fuel.temperature = config['fuel_temp']
    fuel.depletable = True

    gap = openmc.Material(name="gap")
    gap.add_element("He", 1.0)
    gap.set_density("g/cc", 0.000178)
    gap.temperature = config['fuel_temp']

    clad = openmc.Material(name="clad")
    clad.add_element("Zr", 1)
    clad.set_density("g/cc", 6.56)
    clad.temperature = config['clad_temp']

    water = openmc.Material(name="water")
    water.add_element("H", 0.111894, 'wo')
    water.add_element("O", 0.888106, 'wo')
    water.set_density('g/cm3', config['mod_density'])
    water.add_s_alpha_beta("c_H_in_H2O")
    water.temperature = config['mod_temp']

    return fuel, gap, clad, water


def set_material_volumes_quarter(fuel, gap, clad, water, radii, pitch):
    """Set material volumes for quarter-pin geometry.

    radii = [fuel_or, gap_or, clad_or]
    """
    fuel.volume  = math.pi * radii[0]**2 / 4.0
    gap.volume   = math.pi * (radii[1]**2 - radii[0]**2) / 4.0
    clad.volume  = math.pi * (radii[2]**2 - radii[1]**2) / 4.0
    water.volume = (pitch / 2.0)**2 - math.pi * radii[2]**2 / 4.0


def create_quarter_geometry(materials, radii, pitch):
    """Build quarter-pin cell geometry with reflective symmetry.

    Four concentric regions: fuel | He gap | cladding | water
    radii = [fuel_or, gap_or, clad_or]
    """
    fuel_mat, gap_mat, clad_mat, water_mat = materials

    fuel_or = openmc.ZCylinder(r=radii[0], name='fuel_outer')
    gap_or  = openmc.ZCylinder(r=radii[1], name='gap_outer')
    clad_or = openmc.ZCylinder(r=radii[2], name='clad_outer')

    half_pitch = pitch / 2.0
    x_min = openmc.XPlane(x0=0.0,       boundary_type='reflective', name='sym_x')
    x_max = openmc.XPlane(x0=half_pitch, boundary_type='reflective', name='lat_x')
    y_min = openmc.YPlane(y0=0.0,       boundary_type='reflective', name='sym_y')
    y_max = openmc.YPlane(y0=half_pitch, boundary_type='reflective', name='lat_y')

    bounding_box = +x_min & -x_max & +y_min & -y_max

    fuel_cell  = openmc.Cell(name='fuel',  fill=fuel_mat,
                             region=-fuel_or & bounding_box)
    gap_cell   = openmc.Cell(name='gap',   fill=gap_mat,
                             region=+fuel_or & -gap_or & bounding_box)
    clad_cell  = openmc.Cell(name='clad',  fill=clad_mat,
                             region=+gap_or & -clad_or & bounding_box)
    water_cell = openmc.Cell(name='water', fill=water_mat,
                             region=+clad_or & bounding_box)

    root_universe = openmc.Universe(cells=[fuel_cell, gap_cell, clad_cell, water_cell])
    geometry = openmc.Geometry(root_universe)
    return geometry


def create_settings(config):
    """Create OpenMC settings for quarter-pin geometry."""
    settings = openmc.Settings()
    settings.particles = config['particles']
    settings.inactive  = config['inactive']
    settings.batches   = config['batches']
    settings.verbosity = 1

    settings.output = {'tallies': True}

    source = openmc.IndependentSource()
    source.space  = openmc.stats.Point((0.05, 0.05, 0))
    source.angle  = openmc.stats.Isotropic()
    source.energy = openmc.stats.Watt()
    settings.source = source

    settings.temperature = {"method": config.get('temp_method', 'interpolation')}

    if config.get('seed') is not None:
        settings.seed = config['seed']

    return settings


def create_tallies(fuel):
    """Create tallies for flux and reaction rates in the fuel.

    Uses high tally IDs (9001+) to avoid conflicts with the depletion
    operator's internal tallies.

    Returns an openmc.Tallies object with:
      - 'fuel_flux':     total neutron flux in fuel
      - 'fission_rates': fission rate per nuclide
      - 'capture_rates': (n,gamma) capture rate for U238
    """
    tallies = openmc.Tallies()
    mat_filter = openmc.MaterialFilter(fuel)

    t_flux = openmc.Tally(tally_id=9001, name='fuel_flux')
    t_flux.filters = [mat_filter]
    t_flux.scores = ['flux']
    tallies.append(t_flux)

    t_fission = openmc.Tally(tally_id=9002, name='fission_rates')
    t_fission.filters = [mat_filter]
    t_fission.nuclides = FISSION_NUCLIDES
    t_fission.scores = ['fission']
    tallies.append(t_fission)

    t_capture = openmc.Tally(tally_id=9003, name='capture_rates')
    t_capture.filters = [mat_filter]
    t_capture.nuclides = CAPTURE_NUCLIDES
    t_capture.scores = ['(n,gamma)']
    tallies.append(t_capture)

    return tallies