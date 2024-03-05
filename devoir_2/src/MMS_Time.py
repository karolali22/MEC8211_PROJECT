"""

Group Project
MEC8211 - Homework 1
1-D Polar Diffusion PDE with Constant Diffusion Coefficient
1st-Order Forward Time - 2nd-Order Center Space
Neumann BC at r=0 and Dirichlet BC at r=R

"""

import pickle
import numpy as np
import sys
from scipy.linalg import lu_factor, lu_solve

# Spline ----------------------------------------------------------------------

with open('spline_MMS.pkl', 'rb') as file:
    spline = pickle.load(file)

print()
print("Spline object has been loaded.")

# Physical Parameters ---------------------------------------------------------

D = 10**(-2) # Diffusion coefficient in m^2.s^-1
k = 4*10**(-9) # Reaction constant in s^-1
Ce = 12 # Dirichlet boundary constant concentration in mol.m^-3

# Discretization Parameters ---------------------------------------------------

d_r = 0.125/512 # Polar spatial step in m
d_t = 1/512 # Time step in s
R = 0.5 # Polar domain size in m
I = int(round((R + d_r)/d_r, 1)) # Number of spatial steps

# Simulation Parameters -------------------------------------------------------

tolerance = 1e-18  # Define convergence tolerance
max_iterations = int(round((10)/d_t, 1))

print(f"Total Iterations: {max_iterations}")
print()

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

# Source Term -----------------------------------------------------------------

A = -2
lam = 1

def source_term(r, t, D, k, lam, A):
    return (k - lam)*np.exp(-lam*t) - 9*D*A*r + k*A*r**3

# LU Decomposition ------------------------------------------------------------

lu, piv = lu_factor(matrix)

# Simulation ------------------------------------------------------------------

# Initialize C_n with initial conditions
C_n = np.zeros(I)  # Concentration at time t = 0
C_n[0] = 0   # Apply Neumann BC at r = 0
C_n[-1] = Ce  # Apply Dirichlet BC at r = R
iteration = 0  # Keep track of the number of iterations
converged = False  # Flag to check convergence
current_time = 0  # Initialize current time

# To store C_n at each time step
C_n_storage = []
time_steps = []

# Function for displaying progress
def convergence_progress(current_diff, tolerance, iteration):
    percent = 100 * (1 - current_diff / tolerance)
    percent = min(max(percent, 0), 100)
    bar_length = 25
    filled_length = int(bar_length * percent // 100)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\rIteration {iteration}: Convergence [{bar}] {percent:.2f}% (Diff: {current_diff:.2e}, Tolerance: {tolerance:.2e})')
    sys.stdout.flush()

while not converged and iteration < max_iterations:
    rhs = C_n.copy()
    
    # Compute source term
    for i in range(1, I-1):
        r_i = i * d_r
        rhs[i] += source_term(r_i, current_time, D, k, lam, A) * d_t
        
    rhs[0] = 0  # Apply Neumann BC at r = 0
    rhs[-1] = Ce  # Apply Dirichlet BC at r = R

    # Solve M * C_{n+1} = C_{n} using LU decomposition
    C_n_plus_1 = lu_solve((lu, piv), rhs)

    # Check convergence
    diff = np.max(np.abs(C_n_plus_1 - C_n))
    if diff < tolerance:
        converged = True
    else:
        C_n = C_n_plus_1.copy()

    # Save a copy of C_n and the current time for each iteration
    C_n_storage.append(C_n.copy())
    time_steps.append(current_time)

    current_time += d_t
    iteration += 1
    convergence_progress(diff, tolerance, iteration)

print()

if converged:
    print(f"\nConverged after {iteration} iterations.")
else:
    print(f"\nStopped after reaching the maximum number of iterations: {max_iterations}.")

print()

C_n_storage_array = np.array(C_n_storage)

# Error -----------------------------------------------------------------------

L1_errors = []
L2_errors = []
Linf_errors = []

r_values = np.arange(0, R+d_r, d_r)

for t_idx, t in enumerate(time_steps):
    for r_idx, r in enumerate(r_values):
        C_actual = C_n_storage_array[t_idx, r_idx]
        C_pred = spline(t, r)[0, 0]
        error = abs(C_pred - C_actual)
        L1_errors.append(error)
        L2_errors.append(error**2)
        Linf_errors.append(error)

mean_L1_error = np.mean(L1_errors)
mean_L2_error = np.sqrt(np.mean(L2_errors))
mean_Linf_error = np.max(Linf_errors)

print(f"Mesh Nodes I: {I}")
print(f"Mesh Size dr: {d_r}")
print(f"Total Iterations: {max_iterations}")
print(f"Time Step dt: {d_t}")
print(f"Mean L1 Error: {mean_L1_error}")
print(f"Mean L2 Error: {mean_L2_error}")
print(f"Mean Linf Error: {mean_Linf_error}")

