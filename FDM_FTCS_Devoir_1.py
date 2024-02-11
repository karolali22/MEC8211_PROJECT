"""

Simon
MEC8211 - Homework 1
1-D Polar Diffusion PDE with Constant Diffusion Coefficient
1st-Order Forward Time - 2nd-Order Center Space
Neumann BC at r=0 and Dirichlet BC at r=R

"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.linalg import lu_factor, lu_solve

# Physical Parameters ---------------------------------------------------------

D = 10**(-2) # Diffusion coefficient in m^2.s^-1
k = 4*10**(-2) # Reaction constant in s^-1
Ce = 12 # Dirichlet boundary constant concentration in mol.m^-3

# Discretization Parameters ---------------------------------------------------

d_r = 0.0125 # Polar spatial step in m
d_t = 2000 # Time step in s

# Simulation Parameters -------------------------------------------------------

R = 0.5 # Polar domain size in m
I = int(R / d_r) + 1 # Number of spatial steps
T = 10000 # Simulation time in s
N = int(T / d_t) + 1 # Number of time iterations

# Matrix Elements -------------------------------------------------------------

"""
First row: [-3 4 -1 0 ... 0 0]
"""

"""
2nd row: [A2 B2 C2 0 ... 0 0]
3rd row: [0 A3 B3 C3 ... 0 0]
i_th row: A_i in row i-1, B_i in row i, C_i in row i+1, zeros elsewhere

A_i = -D*(1/(d_r**2) - 1/(r_i*2*d_r))*d_t
B_i = 1 - (D*(-2/(d_r**2))-k)*d_t
C_i = -D*(1/(d_r**2) + 1/(r_i*2*d_r))*d_t
r_i = i * d_r # Local polar position in m
"""

"""
Last row: [0 0 0 0 ... 0 1]
"""

# Initialize the matrix
matrix = np.zeros((I, I))

# First row, accounting for Neumann BC at r = 0
matrix[0, :3] = [-3, 4, -1]  # Only the first three elements are non-zero

# Intermediate rows
for i in range(1, I-1):
    r_i = i * d_r  # Local polar position in m
    A_i = -D*(1/(d_r**2) - 1/(r_i*2*d_r))*d_t
    B_i = 1 - (D*(-2/(d_r**2))-k)*d_t
    C_i = -D*(1/(d_r**2) + 1/(r_i*2*d_r))*d_t
    
    # Fill the matrix for the i-th row
    matrix[i, i - 1] = A_i  # Subdiagonal element
    matrix[i, i] = B_i  # Diagonal element
    matrix[i, i + 1] = C_i  # Superdiagonal element

# Last row, accounting for Dirichlet BC at r = R
matrix[-1, -1] = 1

# LU Decomposition ------------------------------------------------------------

lu, piv = lu_factor(matrix)

# Simulation ------------------------------------------------------------------

# Initialize C_n with initial conditions
C_n = np.zeros(I)  # C at time t=0
C_n[0] = 0 # Apply Neumann BC at r = 0
C_n[-1] = Ce  # Apply Dirichlet BC at r = R

# Function for displaying progress
def progress_bar(iteration, total, bar_length=50):
    percent = "{0:.2f}".format(100 * (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = '#' * filled_length + '.' * (bar_length - filled_length)
    sys.stdout.write(f'\rProgress: [{bar}] {percent}%')
    sys.stdout.flush()

# Time-stepping loop
for n in range(N):
    rhs = C_n.copy()
    rhs[0] = 0  # Apply Neumann BC at r = 0
    rhs[-1] = Ce  # Apply Dirichlet BC at r = R
    
    # Solve M * C_{n+1} = C_{n} using LU decomposition
    C_n_plus_1 = lu_solve((lu, piv), rhs)
    C_n = C_n_plus_1.copy()
    progress_bar(n + 1, N)

print()

# Plotting --------------------------------------------------------------------

def plot(C, d_r, R, k, D, Ce, title):
    r_positions = np.arange(0, R + d_r, d_r)[:len(C)]
    r_positions_a = np.linspace(0, R, 1000)
    C_a = np.zeros_like(r_positions_a)
    
    for i, r in enumerate(r_positions_a):
        a = 1 - 1/4 * k/D * R**2 * (r**2/R**2 - 1)
        C_a[i] = Ce / a
    
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(dpi=600, figsize=(6, 3))
    plt.plot(r_positions, C, '-o', label='Numerical', color='black')
    plt.plot(r_positions_a, C_a, label='Analytical', color='red',linewidth=2.5)
    plt.title(title)
    plt.xlabel('r (m)')
    plt.ylabel(r'C (mol/m$^3$)')
    plt.legend()
    plt.grid(True)
    plt.show()

    
plot(C_n, d_r, R, k, D, Ce, 'FTCS: Steady-State Concentration vs Polar Position')
