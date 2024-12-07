""" Molecular Dynamics Simulation (Object-Oriented Version)
This code simulates the dynamics of atoms governed by the Lennard-Jones potential at the molecular scale.
Authors: Borui Xu, Lucien Tsai, Yeqi Chu
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set, Dict

class Atom:
    """Class representing an individual atom in the simulation."""
    def __init__(
        self, 
        position: np.ndarray, 
        velocity: np.ndarray,
        mass: float, 
        epsilon: float, 
        sigma: float) -> None:
        
        self.position: np.ndarray = np.array(position, dtype=float)
        self.velocity: np.ndarray = np.array(velocity, dtype=float)
        self.mass: float = mass
        self.epsilon: float = epsilon
        self.sigma: float = sigma
        self.force: np.ndarray = np.zeros(3, dtype=float)
        
class Potential(ABC):
    """Abstract base class for different potentials."""
    def __init__(self, pairwise_interactions: Optional[Set[Tuple[int, int]]] = None) -> None:
        """
        Args:
            pairwise_interactions: Optional set of (i, j) tuples specifying which atom pairs
                                   should interact. If None, all atom pairs interact.
                                   
            Note: here (i, j) is oriented, and if you want both (0,1) and (1,0) to be valid, 
                  you need to have both (0,1) and (1,0) in the set.
        """
        self.pairwise_interactions = pairwise_interactions  # Set of tuples (i, j)

    @abstractmethod
    def compute_forces(self, atoms: List[Atom], box_size: float) -> None:
        """Compute forces acting on all atoms."""
        pass
    
class Constraint(ABC):
    """Abstract base class for constraints."""
    @abstractmethod
    def apply(self, atoms: List[Atom], box_size: float) -> None:
        """Directly modify atoms to enforce constraints."""
        pass
        
class LennardJonesPotential(Potential):
    """A concrete class implementing the Lennard-Jones potential."""
    def compute_forces(self, atoms: List[Atom], box_size: float) -> None:
        # Reset all forces
        for atom in atoms:
            atom.force.fill(0.0)

        num_atoms: int = len(atoms)
        for i in range(num_atoms):
            atom_i: Atom = atoms[i]
            for j in range(num_atoms):
                if i == j:  # Skip self-interaction
                    continue

                # If defined pairwise_interactions，check if (i, j) is in it
                if self.pairwise_interactions is not None and (i, j) not in self.pairwise_interactions:
                    continue

                atom_j: Atom = atoms[j]
                delta: np.ndarray = atom_i.position - atom_j.position
                # Apply minimum image convention for periodic boundaries
                delta -= box_size * np.round(delta / box_size)
                r: float = float(np.linalg.norm(delta))
                if r == 0:
                    continue
                epsilon_ij: float = np.sqrt(atom_i.epsilon * atom_j.epsilon)
                sigma_ij: float = (atom_i.sigma + atom_j.sigma) / 2
                F_mag: float = 4 * epsilon_ij * (
                    (12 * sigma_ij**12 / r**13) - (6 * sigma_ij**6 / r**7)
                )
                force_vector: np.ndarray = (F_mag / r) * delta
                atom_i.force += force_vector
                atom_j.force -= force_vector  # Newton's third law
                
class FixedDistanceConstraint(Constraint):
    """A constraint that fixes certain atom pairs at a given distance r0."""
    def __init__(
        self, 
        fixed_distances: Dict[Tuple[int, int], float], 
        pairwise_interactions: Optional[Set[Tuple[int,int]]] = None
    ) -> None:
        """
        fixed_distances: {(i,j): r0} means that the target distance of an atom pair (i,j) is r0. 
                         It is usually recommended to also specify (j,i) to make the constraint symmetric. 
        pairwise_interactions: If given, apply this constraint only to these pairs.
        """
        self.fixed_distances = fixed_distances
        self.pairwise_interactions = pairwise_interactions

    def apply(self, atoms: List[Atom], box_size: float) -> None:
        num_atoms = len(atoms)
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue
                if self.pairwise_interactions is not None and (i, j) not in self.pairwise_interactions:
                    continue
                if (i, j) not in self.fixed_distances:
                    continue

                atom_i = atoms[i]
                atom_j = atoms[j]

                delta = atom_i.position - atom_j.position
                # Minimum mirror principle (if periodic boundary conditions are required)
                delta -= box_size * np.round(delta / box_size)
                r = float(np.linalg.norm(delta))
                r0 = self.fixed_distances[(i, j)]

                if r == 0:
                    # If two points coincide, just shift one of them by a random point
                    continue

                # If r! = r0, the position is adjusted
                if r != r0:
                    excess = r - r0
                    direction = delta / r
                    # Assume equal distribution correction
                    correction = direction * (excess / 2.0)

                    # Adjust the position of the atoms so that the final distance changes back to r0
                    atom_i.position -= correction
                    atom_j.position += correction

                    # The coordinates are modified with periodic boundary conditions
                    atom_i.position %= box_size
                    atom_j.position %= box_size
                
class MolecularDynamicsSimulator:
    def __init__(
        self, 
        atoms: List[Atom], 
        box_size: float, 
        total_time: float, 
        total_steps: int,
        potential: Potential,
        constraints: Optional[List[Constraint]] = None
    ) -> None:
        self.atoms = atoms
        self.box_size = box_size
        self.total_time = total_time
        self.total_steps = total_steps
        self.dt = total_time / total_steps
        self.num_atoms = len(atoms)
        self.positions = np.zeros((total_steps + 1, self.num_atoms, 3), dtype=float)
        self.velocities = np.zeros((total_steps + 1, self.num_atoms, 3), dtype=float)
        self.potential = potential
        self.constraints = constraints if constraints is not None else []

    def integrate(self) -> None:
        # Initialize
        for idx, atom in enumerate(self.atoms):
            self.positions[0, idx, :] = atom.position
            self.velocities[0, idx, :] = atom.velocity

        self.potential.compute_forces(self.atoms, self.box_size)

        for step in range(1, self.total_steps + 1):
            # Update positions
            for idx, atom in enumerate(self.atoms):
                atom.position += atom.velocity * self.dt + (atom.force / (2 * atom.mass)) * self.dt**2
                atom.position %= self.box_size

            # Apply constraints
            for c in self.constraints:
                c.apply(self.atoms, self.box_size)

            # Recompute forces after constraint
            self.potential.compute_forces(self.atoms, self.box_size)

            # Update velocities
            for idx, atom in enumerate(self.atoms):
                atom.velocity += (atom.force / atom.mass) * self.dt
                self.positions[step, idx, :] = atom.position
                self.velocities[step, idx, :] = atom.velocity

    def animate(
        self, 
        filename: str = 'md_simulation.gif', 
        interval: int = 50, 
        save_gif: bool = True) -> None:
        
        """Create and optionally save an animation of the simulation."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-self.box_size, self.box_size)
        ax.set_ylim(-self.box_size, self.box_size)
        ax.set_xlabel('X-coordinate (Å)')
        ax.set_ylabel('Y-coordinate (Å)')
        ax.set_title('Molecular Dynamics Simulation')
        scat = ax.scatter([], [], s=100)

        def update(frame: int):
            ax.clear()
            ax.set_xlim(-self.box_size, self.box_size)
            ax.set_ylim(-self.box_size, self.box_size)
            ax.set_xlabel('X-coordinate (Å)')
            ax.set_ylabel('Y-coordinate (Å)')
            ax.set_title(f'Molecular Dynamics Simulation (Time: {frame * self.dt:.2f} ns)')
            positions = self.positions[frame]
            scat = ax.scatter(positions[:, 0], positions[:, 1], s=100, c='blue')
            return scat,

        ani = FuncAnimation(fig, update, frames=range(0, self.total_steps + 1, 10), interval=interval, blit=True)

        if save_gif:
            ani.save(filename, writer='pillow', fps=20)
        plt.show()

# Simulation parameters
box_size: float = 25.0  # Units in Å
total_time: float = 10.0  # Units in ns
total_steps: int = 1000

# Create atoms
initial_positions: np.ndarray = np.array([
    [1.0, 1.0, 1.0],
    [5.0, 5.0, 1.0],
    [10.0, 10.0, 1.0],
    [15.0, 15.0, 1.0],
    [20.0, 20.0, 1.0]
])
initial_velocities: np.ndarray = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
])
mass: float = 1.0  # Units in AMU
epsilon: float = 2.0  # Units in eV
sigma: float = 4.0  # Units in Å

# Create Atom instances
atoms: List[Atom] = [
    Atom(position=pos, velocity=vel, mass=mass, epsilon=epsilon, sigma=sigma)
    for pos, vel in zip(initial_positions, initial_velocities)
]

pairwise_interactions = {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)}
fixed_distances = {
    (0,2): 20.0,
    (2,0): 20.0
}
fixed_distance_constraint = FixedDistanceConstraint(fixed_distances=fixed_distances)

# Create the simulator instance
lj_potential = LennardJonesPotential(pairwise_interactions=pairwise_interactions)
simulator = MolecularDynamicsSimulator(
    atoms=atoms,
    box_size=box_size,
    total_time=total_time,
    total_steps=total_steps,
    potential=lj_potential,
    constraints=[fixed_distance_constraint]
)
# Run simulation
simulator.integrate()

# Animate the results
simulator.animate()
