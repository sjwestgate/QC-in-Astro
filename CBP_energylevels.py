# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:08:42 2026

@author: swest
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh  # for eigenvalues of Hermitian matrices

# Parameters
ratio = 10
E_C = 1.0      # Charging energy (arbitrary units)
E_J = ratio * E_C    # Josephson energy (arbitrary units)
n_levels = 10  # Number of charge states to include
n_g_values = np.linspace(0, 2, 200)  # Gate charge

energy_levels = []

for n_g in n_g_values:
    # Construct Hamiltonian
    dim = 2 * n_levels + 1
    H = np.zeros((dim, dim))
    
    for i in range(dim):
        n = i - n_levels
        H[i, i] = 4 * E_C * (n - n_g)**2  # Charging term
        if i < dim - 1:
            H[i, i+1] = -E_J / 2  # Josephson tunneling
            H[i+1, i] = -E_J / 2
    
    # Compute eigenvalues
    energies = eigh(H, eigvals_only=True)
    energy_levels.append(energies)

energy_levels = np.array(energy_levels)
first_gap = energy_levels[:, 1] - energy_levels[:, 0] #normalise so first energy level=1

# Plot
plt.figure(figsize=(8,6))
for i in range(4):  # plot lowest 4 levels
    plt.plot(n_g_values, energy_levels[:, i]/first_gap[i], label=f'Level {i}')
plt.xlabel('Gate charge $n_g$')
plt.ylabel('Energy')
plt.title('Energy levels')
plt.legend(loc='upper right')
plt.show()