"""

Simon
MEC8211 - Homework 1
1-D Polar Diffusion PDE with Constant Diffusion Coefficient
1st-Order Forward Time - 1st-Order Forward Space
Neumann BC at r=0 and Dirichlet BC at r=R

"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Physical Parameters ---------------------------------------------------------

D = 10**(-10)  # Diffusion coefficient in m^2.s^-1
k = 4**(-9)  # Reaction constant in s^-1
Ce = 12 # Dirichlet boundary constant concentration in mol.m^-3

# Discretization Parameters ---------------------------------------------------

d_r = 0.1  # Polar spatial step in m
d_t = 0.5  # Time step in s

# Simulation Parameters -------------------------------------------------------

R = 0.5  # Polar domain size in m
I = int(R / d_r) + 1 # Number of spatial steps
T = 1000 # Simulation time in s
N = int(T / d_t) + 1 # Number of time iterations

# Matrix Elements -------------------------------------------------------------

"""
First row: [B1 C1 0 0 ... 0 0]

B_1 = 1 + (D*(-2/(d_r**2))-k)*d_t
C_1 = D*(1/(d_r**2))*d_t
"""

"""
2nd row: [A2 B2 C2 0 ... 0 0]
3rd row: [0 A3 B3 C3 ... 0 0]
i_th row: A_i in row i-1, B_i in row i, C_i in row i+1, zeros elsewhere

A_i = D*(1/(d_r**2))*d_t
B_i = 1 + (D*(-2/(d_r**2) - 1/(r_i*d_r))-k)*d_t
C_i = D*(1/(d_r**2) + 1/(r_i*d_r))*d_t
r_i = i * d_r # Local polar position in m
"""

"""
Last row: [0 0 0 0 ... A_I B_I]

A_I = 0
B_I = 1
"""

# Initialize matrix coefficients
A = np.zeros(I)
B = np.zeros(I)
C = np.zeros(I)

# First row, accounting for Neumann BC at r = 0
B[0] = 1 + (D*(-2/(d_r**2))-k)*d_t
C[0] = D*(1/(d_r**2))*d_t

# Intermediate rows
for i in range(1, I-1):
    r_i = i * d_r  # Local polar position in m
    A[i] = D*(1/(d_r**2))*d_t
    B[i] = 1 + (D*(-2/(d_r**2) - 1/(r_i*d_r))-k)*d_t
    C[i] = D*(1/(d_r**2) + 1/(r_i*d_r))*d_t

# Last row, accounting for Dirichlet BC at r = R
B[-1] = 1

# Thomas algorithm function ---------------------------------------------------

def thomas(a, b, c, d):
    I = len(d)  # Number of equations
    # Copy the vectors to avoid modifying the original arrays
    c_prime = np.zeros(I-1)
    d_prime = np.zeros(I)
    
    # Scale factor for the first row
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    # Forward sweep to modify the coefficients
    for i in range(0, I-1):
        scale = 1.0 / (b[i] - a[i] * c_prime[i-1])
        c_prime[i] = c[i] * scale
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) * scale
    
    # Last row
    d_prime[-1] = (d[-1] - a[-1] * d_prime[-2]) / (b[-1] - a[-1] * c_prime[-2])
    
    # Back substitution to find the solution
    x = np.zeros(I)
    x[-1] = d_prime[-1]
    for i in range(I-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x

# Simulation ------------------------------------------------------------------

# Initialize u_n with initial conditions
C_n = np.zeros(I)  # C at time t=0
C_n[-1] = Ce  # Apply Dirichlet BC at r = R

def progress_bar(iteration, total, bar_length=50):
    percent = "{0:.2f}".format(100 * (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = '#' * filled_length + '.' * (bar_length - filled_length)
    sys.stdout.write(f'\rProgress: [{bar}] {percent}%')
    sys.stdout.flush()

# Time-stepping loop
for n in range(N):
    rhs = C_n.copy()
    rhs[-1] = Ce  # Apply Dirichlet BC at r = R
    C_n_plus_1 = thomas(A, B, C, rhs)
    C_n = C_n_plus_1.copy()
    progress_bar(n + 1, N)

print()

# Plotting --------------------------------------------------------------------

def plot(C, d_r, title):
    r_positions = np.arange(0, R + d_r, d_r)[:len(C)]
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(dpi=600,figsize=(6, 3))
    plt.plot(r_positions, C, '-o', color='black')
    plt.title(title)
    plt.xlabel('r (m)')
    plt.ylabel(r'C (mol/m$^3$)')
    plt.grid(True)
    plt.show()
    
plot(C_n, d_r, 'Steady-State Concentration vs Polar Position')
