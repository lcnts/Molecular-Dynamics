# Molecular Dynamics Simulation
## Authors: Borui Xu, Lucien Tsai, Yeqi Chu

## Instructions

All necessary functions and classes are stored in **main.py**.
For ease of usage, **example_cases.ipynb** is provided with the (i) necessary functions and classes and (ii) five example cases (see Structure).

## Structure
1. Abstract Classes
    1. Atom: Defines the properties of the atoms
    3. Potential(ABC): Defines the interaction between atoms.
        - compute_forces: Computes the forces acting on all atoms.
    4. Constraint(ABC): Defines the constraints.
        - apply: Enforces constraints on the relevant atoms.
    5. MolecularDynamicsSimulator
        - integrate: solve the position and velocity with time.
        - animate_positions: Animates the atom positions in 3D.
        - animate_statistics: Animates the kinetic energy and velocity distributions.
        - plot_energy: Plots the total kinetic energy.
3. Concrete Classes
    1. Potential:
        - LennardJonesPotential: Creates a Lennard-Jones potential.
        - LinearSpringPotential: Creates a spring potential for a spring with rest length r0.
    2. Constraint:
        - FixPointConstraint: Creates a constraint that fixes certain atoms to a position.
        - PartialFixConstraint: Creates a constraint that fixes certain atoms in specific Cartesian directions.
        - FixedDistanceConstraint: Creates a constraint that fixes the distance between certain atom pairs at r0.
4. Cases:
    1. Simple Harmonic Oscillator
    2. Three Atoms Under a LJ Potential with Fixed Distance Constraint
    3. Monatomic Molecular Dynamics
    4. Diatomic Molecular Dynamics
    5. Beyond Molecular Dynamics, Tensegrity Structure
