""" Molecular Dynamics Simulation
This code simulations the dynamics of atoms governed by the Lennard-Jones potential at the molecular scale
Authors: Borui Xu, Lucien Tsai, Yeqi Chu
"""

import numpy as np
import matplotlib.pyplot as plt

box_size = 100  # Units in Å
total_time = 10  # Units in ns
total_steps = 2
dt = total_time/total_steps

# Initial Atomic Positions
x_init = np.array([1, 2, 3, 4, 5])
y_init = np.array([1, 2, 3, 4, 5])
z_init = np.array([1, 2, 3, 4, 5])

# Initial Atomic Veolcities
x_dot_init = np.array([0, 0, 0, 0, 0])
y_dot_init = np.array([0, 0, 0, 0, 0])
z_dot_init = np.array([0, 0, 0, 0, 0])

# Atomic Masses
mass = np.array([1, 1, 1, 1, 1])  # Units in AMU

# Lennard-Jones Parameters
well_depth = np.array([2, 2, 2, 2, 2])  # Units in eV
zero_distance = np.array([4, 4, 4, 4, 4])  # Units in Å

def force(atom_index, x, y, z, well_depth, zero_distance):
    """Computes the force on an atom from all other atoms
    Inputs:
        atom_index: index of the atom at which the force is computed
        x: x positions of all atoms
        y: y positions of all atoms
        z: z positions of all atoms
        well_depth: 1st Lennard-Jones parameter of all atoms
        zero_distance: 2nd Lennard-Jones parameter of all atoms
    Outputs:
        F_x: force in x direction
        F_y: force in y direction
        F_z: force in z direction
    """

    F_x = 0
    F_y = 0
    F_z = 0
    for other_index in range(len(x)):
        if other_index != atom_index:
            r = np.sqrt((x[atom_index] - x[other_index]) ** 2 + 
                        (y[atom_index] - y[other_index]) ** 2 + 
                        (z[atom_index] - z[other_index]) ** 2)
            # Compute Lennard-Jones force, below formula to be filled
            # F = ??? 
            F_x += F * (x[atom_index] - x[other_index]) / r
            F_y += F * (y[atom_index] - y[other_index]) / r
            F_z += F * (z[atom_index] - z[other_index]) / r
    return F_x, F_y, F_z


# Compute New Positions and Velocities

# Arrays to store positions at each step for all atoms

def verlet_integration(total_steps,x_init,y_init,z_init,x_all,y_all,z_all,mass, well_depth, zero_distance)
    x_all = np.concatenate((x_init[None, :], np.zeros((total_steps, len(mass)))))
    y_all = np.concatenate((y_init[None, :], np.zeros((total_steps, len(mass)))))
    z_all = np.concatenate((z_init[None, :], np.zeros((total_steps, len(mass)))))

    old_x = x_init
    old_y = y_init
    old_z = z_init

    for i in range(total_steps):  # Verlet Integration (w/o Velocities)
        temp_x = old_x
        temp_y = old_y
        temp_z = old_z
        for j in range(len(mass)):
            F_x, F_y, F_z = force(j, x_all, y_all, z_all, well_depth, zero_distance)
            x_ddot = F_x / mass[j]
            y_ddot = F_y / mass[j]
            z_ddot = F_z / mass[j]

            x_all[i][j] = x_all[j] * 2 - old_x[j] + x_ddot * dt * dt
            y_all[i][j] = y_all[j] * 2 - old_y[j] + y_ddot * dt * dt
            z_all[i][j] = z_all[j] * 2 - old_z[j] + z_ddot * dt * dt
        old_x = temp_x
        old_y = temp_y
        old_z = temp_z
    return x_all,y_all,z_all
