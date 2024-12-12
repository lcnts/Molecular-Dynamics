from script import Atom, Potential, Constraint


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


def test_constraint_initialization():
    """Test if Constraint object can be created with valid parameters."""
    position1 = [0.0, 0.0, 0.0]
    velocity1 = [0.0, 0.0, 0.0]
    mass1 = 1.0
    epsilon1 = 1.0
    sigma1 = 1.0
    atom1 = Atom(position1, velocity1, mass1, epsilon1, sigma1)

    position2 = [1.0, 1.0, 1.0]
    velocity2 = [0.0, 0.0, 0.0]
    mass2 = 1.0
    epsilon2 = 1.0
    sigma2 = 1.0
    atom2 = Atom(position2, velocity2, mass2, epsilon2, sigma2)

    constraint = Constraint(atom1, atom2)
    assert isinstance(constraint, Constraint)
