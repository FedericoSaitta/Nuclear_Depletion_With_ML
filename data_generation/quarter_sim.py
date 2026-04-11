"""
Reactor model setup for BEAVRS-based depletion data generation.

Changes from the original reactor_sim.py:
  - Quarter-pin geometry (reflective at x=0, y=0) for 4x domain reduction
  - Fixed state parameters at BEAVRS Cycle 1 nominal conditions
  - Only power varies between depletion steps
  - Volumes scaled for quarter geometry
"""
import openmc
import math


def update_water_composition(water, boron_ppm, density_g_cm3):
  """Set water composition with given boron concentration and density."""
  h_weight_frac = 0.111894
  o_weight_frac = 0.888106
  boron_weight_frac = boron_ppm * 1e-6

  water.remove_element('H')
  water.remove_element('O')
  if any(elem[0] == 'B' for elem in water.get_elements()):
    water.remove_element('B')

  if boron_weight_frac > 0:
    water.add_element("H", h_weight_frac * (1 - boron_weight_frac), 'wo')
    water.add_element("O", o_weight_frac * (1 - boron_weight_frac), 'wo')
    water.add_element('B', boron_weight_frac, 'wo')
  else:
    water.add_element("H", h_weight_frac, 'wo')
    water.add_element("O", o_weight_frac, 'wo')

  water.set_density('g/cm3', density_g_cm3)


def create_materials(config):
  """Create fuel, cladding, and water materials with BEAVRS specifications.
  
  State parameters are fixed at nominal BEAVRS Cycle 1 hot full power conditions
  rather than being randomly sampled.

  Returns fuel, gap, clad, water (4 materials).

  """
  enrichment = config['enrichment']
  fuel_density = config['fuel_density']

  # --- Fuel (UO2) ---
  fuel = openmc.Material(name="uo2")
  fuel.add_element("U", 1, percent_type="ao", enrichment=enrichment)
  fuel.add_element("O", 2)
  fuel.set_density("g/cc", fuel_density)
  fuel.temperature = config['fuel_temp']
  fuel.depletable = True

  # --- Helium gap ---
  gap = openmc.Material(name="gap")
  gap.add_element("He", 1.0)
  gap.set_density("g/cc", 0.000178)  # He at ~1 atm, ~600K
  gap.temperature = config['fuel_temp']  # gap temp ~ fuel temp

  # --- Cladding (Zircaloy) ---
  clad = openmc.Material(name="clad")
  clad.add_element("Zr", 1)
  clad.set_density("g/cc", 6)
  clad.temperature = config['clad_temp']

  # --- Moderator (borated water) ---
  water = openmc.Material(name="water")
  water.add_element("H", 1.0, 'wo')  # placeholder, replaced below
  update_water_composition(water, config['boron_ppm'], config['mod_density'])
  water.add_s_alpha_beta("c_H_in_H2O")
  water.temperature = config['mod_temp']

  return fuel, gap, clad, water


def set_material_volumes_quarter(fuel, gap, clad, water, radii, pitch):
  """Set material volumes for quarter-pin geometry.
  
  The quarter domain covers [0, pitch/2] x [0, pitch/2], so all
  cylindrical volumes are divided by 4 and the square cell area
  is (pitch/2)^2 instead of pitch^2.
  """
  fuel.volume  = math.pi * radii[0]**2 / 4.0
  gap.volume   = math.pi * (radii[1]**2 - radii[0]**2) / 4.0
  clad.volume  = math.pi * (radii[1]**2 - radii[0]**2) / 4.0
  water.volume = (pitch / 2.0)**2 - math.pi * radii[1]**2 / 4.0


def create_quarter_geometry(materials, radii, pitch):
  """Build quarter-pin cell geometry with reflective symmetry.
  
  Domain: x in [0, pitch/2], y in [0, pitch/2]
  All four boundaries are reflective:
    - x=0 and y=0 are symmetry planes (quarter-pin reduction)
    - x=pitch/2 and y=pitch/2 are lattice periodicity (infinite array)
  
  This is physically equivalent to the full pin cell but with 4x 
  fewer neutrons needed for the same statistical quality.
  """
  fuel_mat, gap_mat, clad_mat, water_mat = materials

  # Cylindrical surfaces (centred at origin)
  fuel_or = openmc.ZCylinder(r=radii[0], name='fuel_outer')
  gap_or  = openmc.ZCylinder(r=radii[1], name='gap_outer')
  clad_or = openmc.ZCylinder(r=radii[1], name='clad_outer')

  # Quarter-domain bounding planes
  half_pitch = pitch / 2.0
  x_min = openmc.XPlane(x0=0.0,    boundary_type='reflective', name='sym_x')
  x_max = openmc.XPlane(x0=half_pitch,  boundary_type='reflective', name='lat_x')
  y_min = openmc.YPlane(y0=0.0,    boundary_type='reflective', name='sym_y')
  y_max = openmc.YPlane(y0=half_pitch,  boundary_type='reflective', name='lat_y')

  bounding_box = +x_min & -x_max & +y_min & -y_max

  # Cells
  fuel_cell  = openmc.Cell(name='fuel',  fill=fuel_mat, region=-fuel_or & bounding_box)
  gap_cell   = openmc.Cell(name='gap',   fill=gap_mat, region=+fuel_or & -gap_or & bounding_box)
  clad_cell  = openmc.Cell(name='clad',  fill=clad_mat, region=+fuel_or & -clad_or & bounding_box)
  water_cell = openmc.Cell(name='water', fill=water_mat, region=+clad_or & bounding_box)

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
  settings.output  = {'tallies': False}

  # Source inside the fuel quarter — slight offset from origin to avoid
  # placing the initial particle exactly on the reflective boundary
  source = openmc.IndependentSource()
  source.space  = openmc.stats.Point((0.05, 0.05, 0))
  source.angle  = openmc.stats.Isotropic()
  source.energy = openmc.stats.Watt()
  settings.source = source

  settings.temperature = {"method": config.get('temp_method', 'interpolation')}

  if config.get('seed') is not None:
    settings.seed = config['seed']

  return settings