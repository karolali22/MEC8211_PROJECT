import numpy as np
import matplotlib.pyplot as plt

h_values_real = np.array([0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625])

# Updated L1 error values
l1_error_values_real = np.array([
    0.0007483735709516992, 0.0008033183502727957, 0.0008117043967305779,
    0.0008142331555840365, 0.000813708330176155, 0.0008134807332173554
])

# Updated L2 error values
l2_error_values_real = np.array([
    0.0009737497998835533, 0.0010019873462728125, 0.0010030639146361951,
    0.0009986682762393045, 0.000995139312921891, 0.0009930366190515339
])

# Updated Linf error values
linf_error_values_real = np.array([
    0.0016737347251432055, 0.0017510709621770815, 0.001779059947564221,
    0.0017867198136993778, 0.0017886838354961299, 0.001789178366271571
])

# Function to fit the last 3 data points of each error to a power law
def fit_power_law(h_values, error_values):
    # Selecting the last 3 data points
    coefficients = np.polyfit(np.log(h_values[-3:]), np.log(error_values[-3:]), 1)
    exponent = coefficients[0]
    fit_function_log = lambda x: exponent * x + coefficients[1]
    return lambda x: np.exp(fit_function_log(np.log(x))), exponent, np.exp(coefficients[1])

fit_function_l1, exponent_l1, prefactor_l1 = fit_power_law(h_values_real, l1_error_values_real)
fit_function_l2, exponent_l2, prefactor_l2 = fit_power_law(h_values_real, l2_error_values_real)
fit_function_linf, exponent_linf, prefactor_linf = fit_power_law(h_values_real, linf_error_values_real)

# Plotting the data and the fits
plt.figure(dpi=600, figsize=(8, 6))
plt.scatter(h_values_real, l1_error_values_real, marker='^', color='g', label='$L_1$ Numerical Values')
plt.plot(h_values_real, fit_function_l1(h_values_real), linestyle='--', color='g', label='$L_1$ Power Law Regression')
plt.scatter(h_values_real, l2_error_values_real, marker='o', color='b', label='$L_2$ Numerical Values')
plt.plot(h_values_real, fit_function_l2(h_values_real), linestyle='--', color='b', label='$L_2$ Power Law Regression')
plt.scatter(h_values_real, linf_error_values_real, marker='s', color='c', label='$L_{\infty}$ Numerical Values')
plt.plot(h_values_real, fit_function_linf(h_values_real), linestyle='--', color='c', label='$L_{\infty}$ Power Law Regression')

plt.title('FTCS: Order of Convergence of $L_1$, $L_2$, and $L_{\infty}$ Errors\nin Function of Mesh Size $Δr$',
          fontsize=14, fontweight='bold', y=1.02)

plt.xlabel('Mesh Size $Δr$ (m)', fontsize=12, fontweight='bold')
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
plt.legend(loc='best')
plt.show()
