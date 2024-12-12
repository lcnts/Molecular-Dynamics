""" Molecular Dynamics Simulation (Object-Oriented Version)
This code simulates the dynamics of atoms governed by the
Lennard-Jones and spring potential at the atomic and lm scale.
Authors: Borui Xu, Lucien Tsai, Yeqi Chu
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D
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
        sigma: float,
    ) -> None:

        self.position: np.ndarray = np.array(position, dtype=float)
        self.velocity: np.ndarray = np.array(velocity, dtype=float)
        self.mass: float = mass
        self.epsilon: float = epsilon
        self.sigma: float = sigma
        self.force: np.ndarray = np.zeros(3, dtype=float)


class Potential(ABC):
    """Abstract base class for different potentials."""

    def __init__(
        self, pairwise_interactions: Optional[Set[Tuple[int, int]]] = None
    ) -> None:
        """
        Args:
            pairwise_interactions:
            Optional set of (i, j) tuples specifying which atom pairs
            should interact. If None, all atom pairs interact.

            Note: here (i, j) is oriented,
            and if you want both (0,1) and (1,0) to be valid,
            you need to have both (0,1) and (1,0) in the set.
        """
        self.pairwise_interactions = pairwise_interactions
        # Set of tuples (i, j)

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
        # Set all forces to 0
        for atom in atoms:
            atom.force.fill(0.0)

        num_atoms: int = len(atoms)
        for i in range(num_atoms):
            atom_i: Atom = atoms[i]
            for j in range(num_atoms):
                if i == j:  # Skip self-interaction
                    continue

                # For defined pairwise_interactions，check if (i, j) is in it
                if (
                    self.pairwise_interactions is not None
                    and (i, j) not in self.pairwise_interactions
                ):
                    continue

                atom_j: Atom = atoms[j]
                delta: np.ndarray = atom_i.position - atom_j.position
                # Apply Periodic Boundary Conditions
                delta -= box_size * np.round(delta / box_size)
                r: float = float(np.linalg.norm(delta))
                if r == 0:
                    continue
                epsilon_ij: float = np.sqrt(atom_i.epsilon * atom_j.epsilon)
                sigma_ij: float = (atom_i.sigma + atom_j.sigma) / 2
                F_mag: float = (
                    4
                    * epsilon_ij
                    * ((12 * sigma_ij**12 / r**13) - (6 * sigma_ij**6 / r**7))
                )
                force_vector: np.ndarray = (F_mag / r) * delta
                atom_i.force += force_vector
                atom_j.force -= force_vector  # Newton's third law


class LinearSpringPotential(Potential):
    """A linear spring potential between pairs of atoms.

    When the distance between two atoms
    equals the rest length r0, the force is zero.
    If the distance deviates from r0,
    a linear restoring force acts on both atoms.
    """

    def __init__(
        self,
        rest_lengths: Dict[Tuple[int, int], float],
        k: float = 1.0,
        pairwise_interactions: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__(pairwise_interactions)
        self.rest_lengths = rest_lengths
        self.k = k

    def compute_forces(self, atoms: List[Atom], box_size: float) -> None:
        # Set all forces to 0
        for atom in atoms:
            atom.force.fill(0.0)

        num_atoms = len(atoms)
        for i in range(num_atoms):
            atom_i = atoms[i]
            for j in range(i + 1, num_atoms):
                atom_j = atoms[j]

                # For defined pairwise_interactions，check if (i, j) is in it
                if self.pairwise_interactions is not None:
                    if (i, j) not in self.pairwise_interactions and (
                        j,
                        i,
                    ) not in self.pairwise_interactions:
                        continue

                # Check if a rest length is defined
                if (i, j) in self.rest_lengths:
                    r0 = self.rest_lengths[(i, j)]
                elif (j, i) in self.rest_lengths:
                    r0 = self.rest_lengths[(j, i)]
                else:
                    continue

                delta = atom_i.position - atom_j.position
                delta -= box_size * np.round(delta / box_size)
                r = float(np.linalg.norm(delta))
                if r == 0:
                    # Skip if atomic positions are the same
                    continue

                # Compute displacement
                dr = r - r0
                if dr != 0.0:
                    # Comppute Force
                    F_mag = self.k * dr
                    # Compute Direction
                    direction = delta / r
                    force_vector = F_mag * direction
                    # Apply Newton's 3rd Law
                    atom_i.force -= force_vector
                    atom_j.force += force_vector


class FixPointConstraint(Constraint):
    """A constraint that fixes certain atoms at a given spatial position."""

    def __init__(
            self, fixed_points: Dict[int, Tuple[float, float, float]]
            ) -> None:
        """
        fixed_points: {atom_index: (x_fixed, y_fixed, z_fixed)}
        """
        self.fixed_points = fixed_points

    def apply(self, atoms: List[Atom], box_size: float) -> None:
        for idx, atom in enumerate(atoms):
            if idx in self.fixed_points:
                # Sets the atomic position to the specified coordinates
                fixed_pos = self.fixed_points[idx]
                atom.position[:] = fixed_pos
                atom.velocity.fill(0.0)


class PartialFixConstraint(Constraint):
    """A constraint that fixes certain
    atoms in specific coordinate directions.

    partial_fixed_points: {atom_index: (x_val, y_val, z_val)}
      - x_val, y_val, z_val can be None or float.
      - If float, that coordinate of the atom is fixed to this value.
      - If None, that coordinate is not constrained.
    """

    def __init__(
        self,
        partial_fixed_points: Dict[
            int, Tuple[Optional[float], Optional[float], Optional[float]]
        ],
    ) -> None:
        self.partial_fixed_points = partial_fixed_points

    def apply(self, atoms: List[Atom], box_size: float) -> None:
        for idx, atom in enumerate(atoms):
            if idx in self.partial_fixed_points:
                x_val, y_val, z_val = self.partial_fixed_points[idx]

                # Constrain x coordinate if specified
                if x_val is not None:
                    atom.position[0] = x_val
                    atom.velocity[0] = 0.0
                    # Set velocity to zero to prevent drift

                # Constrain y coordinate if specified
                if y_val is not None:
                    atom.position[1] = y_val
                    atom.velocity[1] = 0.0

                # Constrain z coordinate if specified
                if z_val is not None:
                    atom.position[2] = z_val
                    atom.velocity[2] = 0.0


class FixedDistanceConstraint(Constraint):
    """A constraint that fixes certain atom pairs at a given distance r0."""

    def __init__(
        self,
        fixed_distances: Dict[Tuple[int, int], float],
        pairwise_interactions: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        """
        fixed_distances: {(i,j): r0} means that
        the target distance of an atom pair (i,j) is r0.
        It is usually recommended to also specify
        (j,i) to make the constraint symmetric.
        pairwise_interactions: If given,
        apply this constraint only to these pairs.
        """
        self.fixed_distances = fixed_distances
        self.pairwise_interactions = pairwise_interactions

    def apply(self, atoms: List[Atom], box_size: float) -> None:
        num_atoms = len(atoms)
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue
                if (
                    self.pairwise_interactions is not None
                    and (i, j) not in self.pairwise_interactions
                ):
                    continue
                if (i, j) not in self.fixed_distances:
                    continue

                atom_i = atoms[i]
                atom_j = atoms[j]

                delta = atom_i.position - atom_j.position
                # Apply Periodic Boundary Conditions
                delta -= box_size * np.round(delta / box_size)
                r = float(np.linalg.norm(delta))
                r0 = self.fixed_distances[(i, j)]

                if r == 0:
                    # Skip if atomic positions are the same
                    continue

                # If r! = r0, the position is adjusted
                if r != r0:
                    excess = r - r0
                    direction = delta / r
                    # Assume equal distribution correction
                    correction = direction * (excess / 2.0)

                    # Adjust the position of the atoms
                    # so that the final distance changes back to r0
                    atom_i.position -= correction
                    atom_j.position += correction

                    # The coordinates are modified
                    # with Periodic Boundary Conditions
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
        constraints: Optional[List[Constraint]] = None,
    ) -> None:
        self.atoms = atoms
        self.box_size = box_size
        self.total_time = total_time
        self.total_steps = total_steps
        self.dt = total_time / total_steps
        self.num_atoms = len(atoms)
        self.positions = np.zeros((total_steps + 1, self.num_atoms, 3),
                                  dtype=float)
        self.velocities = np.zeros((total_steps + 1, self.num_atoms, 3),
                                   dtype=float)
        self.potential = potential
        self.constraints = constraints if constraints is not None else []

    def integrate(self) -> None:
        # Initialize positions and velocities
        for idx, atom in enumerate(self.atoms):
            self.positions[0, idx, :] = atom.position
            self.velocities[0, idx, :] = atom.velocity

        # Compute initial forces
        self.potential.compute_forces(self.atoms, self.box_size)

        # Time integration loop
        for step in range(1, self.total_steps + 1):
            # Update positions using Verlet Integration
            for _idx, atom in enumerate(self.atoms):
                atom.position += (
                    atom.velocity * self.dt
                    + (atom.force / (2 * atom.mass)) * self.dt**2
                )
                atom.position %= self.box_size

            # Apply constraints
            for c in self.constraints:
                c.apply(self.atoms, self.box_size)

            # Compute new forces
            self.potential.compute_forces(self.atoms, self.box_size)

            # Update velocities
            for idx, atom in enumerate(self.atoms):
                atom.velocity += (atom.force / atom.mass) * self.dt
                self.positions[step, idx, :] = atom.position
                self.velocities[step, idx, :] = atom.velocity

    def animate(
        self,
        filename: str = "md_simulation.gif",
        interval: int = 50,
        save_gif: bool = True,
        elev: float = 30,
        azim: float = 60,
        lj_pairs: Optional[List[Tuple[int, int]]] = None,
        fixed_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        # If no pairs are given, default to empty lists
        if lj_pairs is None:
            lj_pairs = []
        if fixed_pairs is None:
            fixed_pairs = []

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Set initial view
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)
        ax.set_zlim(0, self.box_size)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        ax.set_title("Molecular Dynamics Simulation - 3D View")

        # Initial scatter
        # frame_positions = self.positions[0]
        # scat = ax.scatter(
        #     frame_positions[:, 0],
        #     frame_positions[:, 1],
        #     frame_positions[:, 2],
        #     s=100,
        #     c="blue",
        # )

        def update(frame: int):
            ax.clear()
            ax.set_xlim(0, self.box_size)
            ax.set_ylim(0, self.box_size)
            ax.set_zlim(0, self.box_size)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("X (Å)")
            ax.set_ylabel("Y (Å)")
            ax.set_zlabel("Z (Å)")
            ax.set_title(
                f"Molecular Dynamics Simulation "
                f"(Time: {frame * self.dt:.2f} ns)"
            )

            current_positions = self.positions[frame]
            scat = ax.scatter(
                current_positions[:, 0],
                current_positions[:, 1],
                current_positions[:, 2],
                s=100,
                c="blue",
            )

            # Draw LJ potential lines (black dashed)
            for i, j in lj_pairs:
                xi, yi, zi = current_positions[i]
                xj, yj, zj = current_positions[j]
                ax.plot([xi, xj], [yi, yj], [zi, zj], "k--")

            # Draw fixed distance lines (red solid)
            for i, j in fixed_pairs:
                xi, yi, zi = current_positions[i]
                xj, yj, zj = current_positions[j]
                ax.plot([xi, xj], [yi, yj], [zi, zj], "r-")

            return (scat,)

        ani = FuncAnimation(
            fig,
            update,
            frames=range(
                0, self.total_steps + 1, max(1, self.total_steps // 100)),
            interval=interval,
            blit=True,
        )

        if save_gif:
            ani.save(filename, writer="pillow", fps=20)
        plt.show()
