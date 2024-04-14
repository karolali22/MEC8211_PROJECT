"""

MEC8211 - Homework 2
1-D Polar Diffusion PDE with Constant Diffusion Coefficient
1st-Order Forward Time - 1st-Order Forward Space
Neumann BC at r=0 and Dirichlet BC at r=R

"""

import numpy as np
import matplotlib.pyplot as plt

# Time steps (dt) for each simulation
h_values_real = np.array([
    1/64,
    1/128,
    1/256,
])

# L1 error values for each provided time step
l1_error_values_real = np.array([
    2.635826,
    1.093368,
    1.742775,
])

# L2 error values for each provided time step
l2_error_values_real = np.array([
    0.946485,
    0.305751,
    0.452875,
])

# Linf error values for each provided time step
linf_error_values_real = np.array([
    0.551407,
    0.173374,
    0.140390,
])


def fit_power_law(h_values, error_values, start=0, end=2):
    selected_h_values = h_values[start:end+1]
    selected_error_values = error_values[start:end+1]
    coefficients = np.polyfit(np.log(selected_h_values), np.log(selected_error_values), 1)
    exponent = coefficients[0]
    constant = np.exp(coefficients[1])
    fit_func = lambda h: constant * h ** exponent
    return fit_func, exponent, constant

fit_function_l1, exponent_l1, prefactor_l1 = fit_power_law(h_values_real, l1_error_values_real)
fit_function_l2, exponent_l2, prefactor_l2 = fit_power_law(h_values_real, l2_error_values_real)
fit_function_linf, exponent_linf, prefactor_linf = fit_power_law(h_values_real, linf_error_values_real)

# Plotting the data and the fits
plt.figure(figsize=(8, 6))
plt.scatter(h_values_real, l1_error_values_real, marker='^', color='g', label='$L_1$ Numerical Values')
plt.plot(h_values_real, fit_function_l1(h_values_real), linestyle='--', color='g', label='$L_1$ Power Law Regression')
plt.scatter(h_values_real, l2_error_values_real, marker='o', color='b', label='$L_2$ Numerical Values')
plt.plot(h_values_real, fit_function_l2(h_values_real), linestyle='--', color='b', label='$L_2$ Power Law Regression')
plt.scatter(h_values_real, linf_error_values_real, marker='s', color='c', label='$L_{\infty}$ Numerical Values')
plt.plot(h_values_real, fit_function_linf(h_values_real), linestyle='--', color='c', label='$L_{\infty}$ Power Law Regression')

plt.title('MMS: Order of Convergence of $L_1$, $L_2$, and $L_{\infty}$ Errors\nin Function of Time Step $Δt$',
          fontsize=14, fontweight='bold', y=1.02)

plt.xlabel('Time Step $Δt$ (m)', fontsize=12, fontweight='bold')
plt.ylabel('Error (mol/m$^3$)', fontsize=12, fontweight='bold')

# Styling the plot
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)

# Display equations on the plot
equation_text_l1 = f'$L_1 = {prefactor_l1:.4f} \\times Δr^{{{exponent_l1:.4f}}}$'
equation_text_l2 = f'$L_2 = {prefactor_l2:.4f} \\times Δr^{{{exponent_l2:.4f}}}$'
equation_text_linf = f'$L_{{\infty}} = {prefactor_linf:.4f} \\times Δr^{{{exponent_linf:.4f}}}$'
plt.text(0.03, 0.7, equation_text_l1, fontsize=12, transform=plt.gca().transAxes, color='g')
plt.text(0.03, 0.6, equation_text_l2, fontsize=12, transform=plt.gca().transAxes, color='b')
plt.text(0.03, 0.5, equation_text_linf, fontsize=12, transform=plt.gca().transAxes, color='c')

plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend(loc='lower right')
plt.show()
