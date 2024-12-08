# Molecular Dynamics Method (for tensegrity structure??)
## Authors: Borui Xu, Lucien Tsai, Yeqi Chu

1. Algorithm: Molecular Dynamics Method
2. Inputs: potentials, constraints, initial condition of atoms
3. Outputs: A .gif of the motion of the atoms

All codes are in script.ipynb.

## Structure
1. class
    1. Atom
    2. Potential(ABC): Define the interaction between atoms
        - compute_force: compute the force from the potential
    3. Constraint(ABC): Define the constraints
        - apply: apply the constraint to atoms
    4. MolecularDynamicsSimulator
        - integrate: solve the displacements and velocity with time
        - animate: save the displacements into a .gif
2. concrete class of Potential and Constraint
    1. Potential:
        1. Lennard-Jones
        2. Linear spring
    2. Constraint:
        1. Fix point: fix an atom's x,y,z displacement
        2. Partial fix: fix an atom's displacement, but 1 or 2 or 3 components of it. For example: only fix the z-direction displacement.
        3. Fix distance: fix the distance between two atoms
3. Examples:
    1. 5 points in a line
    2. A kresling-like tensegrity


