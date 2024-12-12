import pytest
from script import Atom

# Basic test case for Atom initialization
def test_atom_basic_initialization():
    """Test if Atom object can be created."""
    atom = Atom()
    assert isinstance(atom, Atom)  # Verify the instance type
