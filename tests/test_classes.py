# import pytest
from script import Atom, Potential


# Mock subclass of Potential for testing
class TestPotential(Potential):
    def compute_forces(self):
        """Mock implementation of abstract method."""
        pass


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
    """Test if Potential object can be created using the mock subclass."""
    potential = TestPotential()  # Removed `cutoff` parameter
    assert isinstance(potential, Potential)  # Verify the instance type
