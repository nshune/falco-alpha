from pathlib import Path
from unittest import TestCase

from falco import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from falco.core.dynamics.axis import Axis, ValidOrigins
from falco.core.loads.forces_moments import Vector
from falco.core.loads.mass_properties import MassProperties, MassMI
from falco.core.vehicle.components.component import Component, ComponentParameters
from falco.utils.import_geometry import import_geometry


class TestComponentInitialization(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_default_initialization(self):
        """Test default initialization of a Component."""
        component = Component(name="custom_component")
        self.assertEqual(component._name, "custom_component")
        self.assertIsNone(component.geometry)
        self.assertFalse(component.compute_surface_area_flag)
        self.assertIsNone(component._parameterization_solver)
        self.assertIsNone(component.mass_properties)
        self.assertIsNone(component._ffd_geometric_variables)
        self.assertIsInstance(component.parameters, ComponentParameters)
        self.assertEqual(component.comps, {})
        self.assertEqual(component.surface_mesh, [])
        self.assertEqual(component.load_solvers, [])
        self.assertIsNone(component.surface_area)


    def test_kwargs_parameters(self):
        """Test initialization with user-defined parameters using kwargs."""
        component = Component(name="custom_component",
                              param1="value1", param2=123)
        self.assertEqual(component.parameters.param1, "value1")
        self.assertEqual(component.parameters.param2, 123)

    def test_parent_attribute(self):
        """Test the parent attribute initialization."""
        component = Component(name="custom_component")
        self.assertIsNone(component.parent)

    def test_empty_subcomponents(self):
        """Test that the subcomponents dictionary is empty upon initialization."""
        component = Component(name="custom_component")
        self.assertEqual(component.comps, {})

    # def test_providing_mp(self):
    #     component = Component(name="custom_component")

        axis = Axis(
            name='Inertial Axis',
            x=np.array([0]) * ureg.meter,
            y=np.array([0]) * ureg.meter,
            z=np.array([0]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value
        )

        # Create a Quantity vector with units
        cg = Q_([0, 0, 0], 'newton')
        cg = Vector(cg, axis)
        # Create a mass moment of inertia object
        mi = MassMI(axis=axis)
        # Create a mass properties object
        mp = MassProperties(cg=cg, inertia=mi, mass=Q_(10, 'lb'))


class TestComponentGeometry(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        from falco import GEOMETRIES_ROOT_FOLDER
        self.wing_geometry = import_geometry("simple_wing.stp", file_path=GEOMETRIES_ROOT_FOLDER)

    def test_geometry_surface_area(self):
        wing_component = Component(name="Wing",
                                   geometry=self.wing_geometry,
                                   compute_surface_area_flag=True)
        np.testing.assert_almost_equal(wing_component.surface_area.value, 94.82, decimal=3)


class TestComponentHierarchy(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_adding_subcomponent(self):
        """Test adding a subcomponent to a Component."""
        parent_component = Component(name="parent_component")
        sub_component = Component(name="sub_component")

        parent_component.add_subcomponent(sub_component)

        self.assertIn("sub_component", parent_component.comps)
        self.assertEqual(parent_component.comps["sub_component"], sub_component)
        self.assertEqual(sub_component.parent, parent_component)

    def test_adding_subcomponent_that_already_exists(self):
        """Test adding a subcomponent that already exists in the parent Component."""
        parent_component = Component(name="parent_component")
        sub_component = Component(name="sub_component")

        parent_component.add_subcomponent(sub_component)

        with self.assertRaises(KeyError):
            parent_component.add_subcomponent(sub_component)

    def test_removing_subcomponent(self):
        """Test removing a subcomponent from a Component."""
        parent_component = Component(name="parent_component")
        sub_component1 = Component(name="sub_component1")
        sub_component2 = Component(name="sub_component2")

        parent_component.add_subcomponent(sub_component1)
        parent_component.add_subcomponent(sub_component2)

        parent_component.remove_subcomponent(sub_component1)

        self.assertNotIn("sub_component1", parent_component.comps)
        self.assertIsNone(sub_component1.parent)
        self.assertIn("sub_component2", parent_component.comps)

    def test_viz_component_hierarchy(self):
        parent_component = Component(name="parent_component")
        sub_component1 = Component(name="sub_component1")
        sub_component2 = Component(name="sub_component2")

        parent_component.add_subcomponent(sub_component1)
        parent_component.add_subcomponent(sub_component2)

        parent_component.visualize_component_hierarchy(filepath=Path.cwd()/"python_test_outputs")
        self.assertTrue((Path.cwd()/"python_test_outputs/component_hierarchy.png").exists())
        # Delete the folder and all its files after the test
        (Path.cwd()/"python_test_outputs/component_hierarchy.png").unlink()
        (Path.cwd()/"python_test_outputs/component_hierarchy").unlink()
        (Path.cwd() / "python_test_outputs/").rmdir()
