""" Molecular Dynamics Simulation (Object-Oriented Version)
This code simulates the dynamics of atoms governed by the Lennard-Jones potential at the molecular scale.
Authors: Borui Xu, Lucien Tsai, Yeqi Chu
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from abc import ABC, abstractmethod
from typing import List

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
    @abstractmethod
    def compute_forces(
        self, 
        atom: List[Atom],
        box_size: float) -> None:
        """Compute forces acting on all atoms.
        
        Args:
            atoms: List of Atom objects in the simulation.
            box_size: Size of the simulation box for periodic boundary conditions.
        """
        pass
    
class LennardJonesPotential(Potential):
    """A concrete class implementing the Lennard-Jones potential."""
    def compute_forces(
        self,
        atoms: List[Atom], 
        box_size: float) -> None:
        # Reset all forces
        for atom in atoms:
            atom.force.fill(0.0)
        
        num_atoms: int = len(atoms)
        for i in range(num_atoms):
            atom_i: Atom = atoms[i]
            for j in range(i + 1, num_atoms):
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
                atom_j.force -= force_vector
                
class MolecularDynamicsSimulator:
    """Class for simulating molecular dynamics using the Lennard-Jones potential."""
    def __init__(
        self, 
        atoms: List[Atom], 
        box_size: float, 
        total_time: float, 
        total_steps: int,
        potential: Potential) -> None:
        
        self.atoms: List[Atom] = atoms
        self.box_size: float = box_size
        self.total_time: float = total_time
        self.total_steps: int = total_steps
        self.dt: float = total_time / total_steps
        self.num_atoms: int = len(atoms)
        self.positions: np.ndarray = np.zeros((total_steps + 1, self.num_atoms, 3))
        self.velocities: np.ndarray = np.zeros((total_steps + 1, self.num_atoms, 3))
        self.potential: Potential = potential 

    def integrate(self) -> None:
        """Perform the Verlet integration over all time steps."""
        # Initialize positions and velocities
        for idx, atom in enumerate(self.atoms):
            self.positions[0, idx, :] = atom.position
            self.velocities[0, idx, :] = atom.velocity

        # Compute initial forces using the assigned potential
        self.potential.compute_forces(self.atoms, self.box_size)

        # Time integration loop
        for step in range(1, self.total_steps + 1):
            for idx, atom in enumerate(self.atoms):
                # Update positions
                atom.position += atom.velocity * self.dt + (atom.force / (2 * atom.mass)) * self.dt ** 2
                # Apply periodic boundary conditions
                atom.position %= self.box_size

            # Compute new forces
            self.potential.compute_forces(self.atoms, self.box_size)

            for idx, atom in enumerate(self.atoms):
                # Update velocities
                atom.velocity += (atom.force / atom.mass) * self.dt
                # Store positions and velocities
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

# Create the simulator instance
lj_potential = LennardJonesPotential()
simulator = MolecularDynamicsSimulator(atoms, box_size, total_time, total_steps, lj_potential)

# Run simulation
simulator.integrate()

# Animate the results
simulator.animate()