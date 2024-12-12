# import pytest
from script import Atom


# Basic test case for Atom initialization
def test_atom_basic_initialization():
    """Test if Atom object can be created with valid inputs."""
    position = [0.0, 0.0, 0.0]
    velocity = [0.0, 0.0, 0.0]
    mass = 1.0
    epsilon = 1.0
    sigma = 1.0

    atom = Atom(position, velocity, mass, epsilon, sigma)
    assert isinstance(atom, Atom)  # Verify the instance type
