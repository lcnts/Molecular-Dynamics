from script import Atom, Potential, LennardJonesPotential
import numpy as np


class TestPotential(Potential):
    def compute_forces(self):
        """Mock implementation of abstract method."""
        pass


def test_atom_basic_initialization():
    """Test if Atom object can be created."""
    position = [0.0, 0.0, 0.0]
    velocity = [0.0, 0.0, 0.0]
    mass = 1.0
    epsilon = 1.0
    sigma = 1.0
    atom = Atom(position, velocity, mass, epsilon, sigma)
    assert isinstance(atom, Atom)


def test_potential_initialization():
    """Test if Potential object can
    be created using the mock subclass."""
    potential = TestPotential()
    assert isinstance(potential, Potential)


def test_lennard_jones_force_computation():
    """Test the force computation for
    two atoms using Lennard-Jones potential."""
    position1 = [0.0, 0.0, 0.0]
    position2 = [1.0, 0.0, 0.0]  # One unit distance apart on the x-axis
    velocity1 = [0.0, 0.0, 0.0]
    velocity2 = [0.0, 0.0, 0.0]
    mass = 1.0
    epsilon = 1.0
    sigma = 1.0

    atom1 = Atom(position1, velocity1, mass, epsilon, sigma)
    atom2 = Atom(position2, velocity2, mass, epsilon, sigma)
    atoms = [atom1, atom2]

    lj_potential = LennardJonesPotential()

    box_size = 10.0
    lj_potential.compute_forces(atoms, box_size)

    assert np.linalg.norm(atom1.force) > 0
    assert np.linalg.norm(atom2.force) > 0

    np.testing.assert_almost_equal(atom1.force, -atom2.force, decimal=5)
