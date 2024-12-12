# import pytest
from script import Atom, Potential


# Basic test case for Atom initialization
def test_atom_basic_initialization():
    """Test if Atom object can be created."""
    position = [0.0, 0.0, 0.0]
    velocity = [0.0, 0.0, 0.0]
    mass = 1.0
    epsilon = 1.0
    sigma = 1.0
    atom = Atom(position, velocity, mass, epsilon, sigma)
    assert isinstance(atom, Atom)  # Verify the instance type


# Basic test case for Potential initialization
def test_potential_initialization():
    """Test if Potential object can be created with parameters."""
    potential = Potential(cutoff=10.0)
    assert isinstance(potential, Potential)  # Verify the instance type
    assert potential.cutoff == 10.0  # Verify the cutoff parameter
