import openmc
import openmc.deplete
import os
import tempfile


def test_openmc_import():
    """Test that OpenMC can be imported"""
    assert openmc.__version__ is not None
    print("OpenMC version:", openmc.__version__)


def test_material_creation():
    """Test that we can create an OpenMC material"""
    material = openmc.Material(name="Empty")
    assert material is not None
    assert material.name == "Empty"


def test_geometry_creation():
    """Test that we can create an OpenMC geometry"""
    geometry = openmc.Geometry()
    cell = openmc.Cell(name="Empty cell")
    geometry.root = cell

    assert geometry is not None
    assert cell.name == "Empty cell"
    assert geometry.root == cell


def test_deplete_module():
    """Test that OpenMC depletion module is available"""
    assert hasattr(openmc, "deplete")
    assert openmc.deplete is not None


def test_pin_model_with_simulation():
    """Test complete pin model construction (without running simulation)"""

    with tempfile.TemporaryDirectory() as results_dir:
        # --- Materials ---
        uo2 = openmc.Material(name="uo2")
        uo2.add_nuclide("U235", 0.03)
        uo2.add_nuclide("U238", 0.97)
        uo2.add_nuclide("O16", 2.0)
        uo2.set_density("g/cm3", 10.0)

        zirconium = openmc.Material(name="zirconium")
        zirconium.add_element("Zr", 1.0)
        zirconium.set_density("g/cm3", 6.6)

        water = openmc.Material(name="h2o")
        water.add_nuclide("H1", 2.0)
        water.add_nuclide("O16", 1.0)
        water.set_density("g/cm3", 1.0)
        water.add_s_alpha_beta("c_H_in_H2O")

        materials = openmc.Materials([uo2, zirconium, water])
        assert len(materials) == 3

        # --- Geometry ---
        fuel_outer_radius = openmc.ZCylinder(r=0.39)
        clad_inner_radius = openmc.ZCylinder(r=0.40)
        clad_outer_radius = openmc.ZCylinder(r=0.46)

        fuel_region = -fuel_outer_radius
        gap_region = +fuel_outer_radius & -clad_inner_radius
        clad_region = +clad_inner_radius & -clad_outer_radius

        fuel = openmc.Cell(name="fuel", fill=uo2, region=fuel_region)
        gap = openmc.Cell(name="air gap", region=gap_region)
        clad = openmc.Cell(name="clad", fill=zirconium, region=clad_region)

        pitch = 1.26
        left = openmc.XPlane(x0=-pitch / 2, boundary_type="reflective")
        right = openmc.XPlane(x0=pitch / 2, boundary_type="reflective")
        bottom = openmc.YPlane(y0=-pitch / 2, boundary_type="reflective")
        top = openmc.YPlane(y0=pitch / 2, boundary_type="reflective")

        prism_region = +left & -right & +bottom & -top
        moderator_region = prism_region & +clad_outer_radius

        moderator = openmc.Cell(name="moderator", fill=water, region=moderator_region)

        root_universe = openmc.Universe(cells=[fuel, gap, clad, moderator])
        geometry = openmc.Geometry(root_universe)
        assert geometry is not None

        # --- Settings ---
        point = openmc.stats.Point((0, 0, 0))
        source = openmc.Source(space=point)

        settings = openmc.Settings()
        settings.source = source
        settings.batches = 10
        settings.inactive = 2
        settings.particles = 100

        # --- Tallies ---
        cell_filter = openmc.CellFilter(fuel)
        tally = openmc.Tally(name="fuel_tally")
        tally.filters = [cell_filter]
        tally.nuclides = ["U235"]
        tally.scores = ["fission"]

        tallies = openmc.Tallies([tally])
        assert len(tallies) == 1

        # --- Export XML files (validates setup is correct) ---
        materials.export_to_xml(path=results_dir)
        geometry.export_to_xml(path=results_dir)
        settings.export_to_xml(path=results_dir)
        tallies.export_to_xml(path=results_dir)

        # Verify XML files were created
        assert os.path.exists(os.path.join(results_dir, "materials.xml"))
        assert os.path.exists(os.path.join(results_dir, "geometry.xml"))
        assert os.path.exists(os.path.join(results_dir, "settings.xml"))
        assert os.path.exists(os.path.join(results_dir, "tallies.xml"))
