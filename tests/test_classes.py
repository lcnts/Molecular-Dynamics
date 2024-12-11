import pytest
from script import Atom, Potential, Constraint, LennardJonesPotential, LinearSpringPotential

# Test for the Atom class
def test_atom_initialization():
    """Test if Atom object is initialized properly."""
    atom = Atom()
    assert isinstance(atom, Atom)  # Check if atom is an instance of Atom

# Test for the Potential class
def test_potential_initialization():
    """Test if Potential object initializes its attributes correctly."""
    potential = Potential()
    assert isinstance(potential, Potential)  # Check if the object is an instance of Potential

# Test for the compute_forces method in Potential class
def test_potential_compute_forces():
    """Test if compute_forces method runs without errors."""
    potential = Potential()
    try:
        result = potential.compute_forces()
        assert result is not None  # Ensure the method returns some output
    except Exception as e:
        pytest.fail(f"compute_forces raised an exception: {e}")

# Test for the Constraint class
def test_constraint_apply():
    """Test if the apply method in Constraint runs without errors."""
    constraint = Constraint()
    try:
        constraint.apply()
    except Exception as e:
        pytest.fail(f"apply method raised an exception: {e}")

# Test for the LennardJonesPotential class
def test_lennard_jones_potential_compute_forces():
    """Test if compute_forces in LennardJonesPotential runs without errors."""
    potential = LennardJonesPotential()
    try:
        result = potential.compute_forces()
        assert result is not None  # Ensure the method returns some output
    except Exception as e:
        pytest.fail(f"compute_forces raised an exception: {e}")

# Test for the LinearSpringPotential class
def test_linear_spring_potential_initialization():
    """Test if LinearSpringPotential initializes correctly."""
    potential = LinearSpringPotential()
    assert isinstance(potential, LinearSpringPotential)  # Check if the object is an instance of LinearSpringPotential

# Test for the compute_forces method in LinearSpringPotential
def test_linear_spring_potential_compute_forces():
    """Test if compute_forces in LinearSpringPotential runs without errors."""
    potential = LinearSpringPotential()
    try:
        result = potential.compute_forces()
        assert result is not None  # Ensure the method returns some output
    except Exception as e:
        pytest.fail(f"compute_forces raised an exception: {e}")
