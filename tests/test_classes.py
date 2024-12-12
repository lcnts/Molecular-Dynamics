from script import Atom, Potential, LennardJonesPotential


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


def test_lennard_jones_initialization():
    """Test if LennardJonesPotential
    object can be created with valid parameters."""
    epsilon = 1.0
    sigma = 1.0
    lj_potential = LennardJonesPotential(epsilon, sigma)
    assert isinstance(lj_potential, LennardJonesPotential)
    assert lj_potential.compute_force(1.0) is not None
