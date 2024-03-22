"""

MEC8211 - Homework 3 - Group 4

"""

import numpy as np
import matplotlib.pyplot as plt

a = 30.455317398900775
b = 25.208961791608992
c = 24.874012898627328
d = 24.756196768165594

f0 = 4/3 * d - 1/3 * c

# Spatial steps (dx) for each simulation
h_values_real = np.array([
    4,
    2,
    1,
    0.5,
])

# L1 error values for each provided time step
error_values_real = np.array([
    abs(a - f0),
    abs(b - f0),
    abs(c - f0),
    abs(d - f0),
])

def fit_power_law(h_values, error_values, start=6, end=8):
    selected_h_values = h_values[start:end+1]
    selected_error_values = error_values[start:end+1]
    coefficients = np.polyfit(np.log(selected_h_values), np.log(selected_error_values), 1)
    exponent = coefficients[0]
    constant = np.exp(coefficients[1])
    fit_func = lambda h: constant * h ** exponent
    return fit_func, exponent, constant

fit_function_l1, exponent_l1, prefactor_l1 = fit_power_law(h_values_real, error_values_real, start=0, end=3)

# Plotting the data and the fits
plt.figure(dpi=600, figsize=(8, 6))
plt.scatter(h_values_real, error_values_real, marker='^', color='black', label='Numerical Values')
plt.plot(h_values_real, fit_function_l1(h_values_real), linestyle='--', color='black', label='Power Law Regression')

plt.title('Order of Convergence: Error in Function\n of Grid Size $Δx$ for Seed 11',
          fontsize=14, fontweight='bold', y=1.02)

plt.xlabel('Grid Size $Δx$ ($\mu$m$^2$)', fontsize=12, fontweight='bold')
plt.ylabel(r'Error ($\mu$m$^2$)', fontsize=12, fontweight='bold')

# Styling the plot
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)

# Display equations on the plot
equation_text_l1 = f'$Error = {prefactor_l1:.4f} \\times Δr^{{{exponent_l1:.4f}}}$'
plt.text(0.03, 0.7, equation_text_l1, fontsize=12, transform=plt.gca().transAxes, color='black')

plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend(loc='lower right')
plt.show()
