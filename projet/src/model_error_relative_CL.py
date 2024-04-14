import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

gci = '../data/CL/num_uncertainty_CL.txt'
setpoint = "../data/CL/distribution_re_5970000.txt"
lower_bound = "../data/CL/distribution_re_5591000.txt"
upper_bound = "../data/CL/distribution_re_6725000.txt"

exp_un = 0.004
k = 2

def read_gci_values(filename):
    gci_values = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split(': ', 1)
            if len(parts) < 2:
                continue
            json_string = parts[1].replace("'", '"')
            try:
                data_dict = json.loads(json_string)
                gci_values.append(data_dict['GCI'])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    return gci_values

def read_column(file_path, column_index):
    return pd.read_csv(file_path, sep="\t", usecols=[column_index]).iloc[:, 0]

gci_values = read_gci_values(gci)
num_un = np.divide(gci_values, 2)

data = pd.read_csv(setpoint, sep="\t", usecols=[0, 1, 3])
x = data.iloc[:, 0]
y = data.iloc[:, 2]

s_dev2 = read_column(lower_bound, 1)
s_dev3 = read_column(upper_bound, 1)

mean_2 = read_column(lower_bound, 3)
mean_3 = read_column(upper_bound, 3)

shift_2 = np.absolute(mean_2 - y)
shift_3 = np.absolute(mean_3 - y)

input_un = np.maximum(s_dev2.values, s_dev3.values) + np.maximum(shift_2, shift_3)

x_values = [-4.05, -2.00, 0.05, 1.98, 4.18, 6.20, 8.22, 10.18, 11.08, 12.25, 13.10, 14.28, 15.20, 16.18, 16.9, 17.35, 17.65, 18.65]
y_values = [-0.4280, -0.2150, 0.0040, 0.2080, 0.4520, 0.6630, 0.8800, 1.0880, 1.1800, 1.2920, 1.3680, 1.4580, 1.5280, 1.5900, 1.6180, 1.6600, 1.6450, 1.0050]

ci_y = k*np.divide(np.sqrt(input_un**2 + num_un**2 + exp_un**2), np.abs(y_values))*100
ci_x = 0.75

y_error = (y - y_values)*100/y_values

# Plotting
plt.figure(figsize=(10, 6))

plt.axhline(y=0, color='black', linewidth=1)

plt.errorbar(x, y_error, yerr=ci_y, xerr=ci_x, fmt='s', ecolor='red', elinewidth=1.5, capsize=2, capthick=2, markersize=4, markerfacecolor='none', mec='red', mew=1, alpha=1)
plt.scatter(x, y_error, marker='s', color='red', facecolor='none', s=20)

plt.xlabel(r'Angle of Attack $\alpha$ [°]', fontsize=12)
plt.ylabel(r'$\delta_{model}=(C_{L_{sim}}-C_{L_{exp}})$ / $C_{L_{exp}}$ [%]', fontsize=12)
plt.ylim(-150, 300)
plt.xlim(-5, 20)
plt.grid( linestyle='--', linewidth=0.5)

plt.savefig('../results/model_error_relative_CL.png', dpi=600)

plt.show()
