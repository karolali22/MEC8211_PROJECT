import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

gci = '../data/CD/num_uncertainty_CD.txt'
setpoint = "../data/CD/distribution_re_5970000.txt"
lower_bound = "../data/CD/distribution_re_5591000.txt"
upper_bound = "../data/CD/distribution_re_6725000.txt"

exp_un = 0.0002
k = 2

def read_gci_values(filename):
    gci_values = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = lines[:-2]
        for line in lines:
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
    return pd.read_csv(file_path, sep="\t", usecols=[column_index]).iloc[:-2, 0]

gci_values = read_gci_values(gci)
num_un = np.divide(gci_values, 2)

data = pd.read_csv(setpoint, sep="\t", usecols=[0, 1, 3])
x = data.iloc[:-2, 0]
y = data.iloc[:-2, 2]

s_dev2 = read_column(lower_bound, 1)
s_dev3 = read_column(upper_bound, 1)

mean_2 = read_column(lower_bound, 3)
mean_3 = read_column(upper_bound, 3)

shift_2 = np.absolute(mean_2 - y)
shift_3 = np.absolute(mean_3 - y)

input_un = np.maximum(s_dev2.values, s_dev3.values) + np.maximum(shift_2, shift_3)

ci_y = k*np.sqrt(input_un**2 + num_un**2 + exp_un**2)
ci_x = 0.75

x_values = [-4.05, -2.00, 0.05, 1.98, 4.18, 6.20, 8.22, 10.18, 11.08, 12.25, 13.10, 14.28, 15.20, 16.18, 16.9, 17.35]
y_values = [0.00700, 0.00650, 0.00650, 0.00680, 0.00760, 0.00680, 0.00800, 0.01050, 0.01140, 0.01250, 0.01300, 0.01620, 0.01870, 0.02180, 0.02440, 0.02750]

# Plotting
plt.figure(figsize=(10, 6))

plt.errorbar(x, y, yerr=ci_y, xerr=ci_x, fmt='s', ecolor='red', elinewidth=1.5, capsize=2, capthick=2, markersize=4, markerfacecolor='none', mec='red', mew=1, alpha=1)
plt.plot(x, y, '--', color='red', alpha=0.5)
plt.scatter(x, y, marker='s', color='red', facecolor='none', s=20, label=r'$C_D$ - Numerical')

plt.scatter(x_values, y_values, marker='D', s=20, label=r'$C_D$ - Experimental', color='black', facecolor='none')

plt.xlabel(r'Angle of Attack $\alpha$ [°]', fontsize=14)
plt.ylabel(r'$C_D$', fontsize=14)
plt.ylim(-0.10, 0.20)
plt.xlim(-5, 20)
plt.legend()
plt.grid(linestyle='--', linewidth=0.5)

plt.savefig('../results/confidence_intervals_CD.png', dpi=600)

plt.show()