from script import Atom, Potential


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
    """Test if Potential object can be created using the mock subclass."""
    potential = TestPotential()
    assert isinstance(potential, Potential)


def test_potential_compute_forces():
    """Test if compute_forces method works in the mock subclass."""
    potential = TestPotential()
    forces = potential.compute_forces()
    assert isinstance(forces, list), f"Expected list, got {type(forces)}"
    assert len(forces) == 3, f"Expected 3 forces, got {len(forces)}"
    assert all(
        force == 0.0 for force in forces), f"Expected all zeros, got {forces}"
