"""

MEC8211 - Homework 3 - Group 4

"""

import numpy as np
from scipy.optimize import fsolve

median = 80.6
std_dev_log_normal = 14.7

mu = np.log(median)

def equation(sigma):
    return (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2) - std_dev_log_normal**2

initial_guess = 0.1

sigma_solution = fsolve(equation, initial_guess)
sigma = sigma_solution[0]

geo_std_dev = np.exp(sigma)

interval_lower = median / geo_std_dev
interval_upper = median * geo_std_dev

print(f"Solved sigma: {sigma}")
print(f"Interval for one standard deviation: [{interval_lower}, {interval_upper}]")
