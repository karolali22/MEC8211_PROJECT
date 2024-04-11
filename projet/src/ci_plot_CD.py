import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "../data/CD/distribution_re_5970000.txt"
file_path2 = "../data/CD/distribution_re_5793000.txt"
file_path3 = "../data/CD/distribution_re_6154000.txt"

def read_column(file_path, column_index):
    return pd.read_csv(file_path, sep="\t", usecols=[column_index]).iloc[:-2, 0]

data2 = read_column(file_path2, 1)
data3 = read_column(file_path3, 1)

max_val = np.maximum(data2.values, data3.values)

data = pd.read_csv(file_path, sep="\t", usecols=[0, 1, 3])
x = data.iloc[:-2, 0]
y = data.iloc[:-2, 2]

ci_y = np.sqrt(max_val**2 + 0.10**2 + 0.0002**2)
ci_x = 0.01

x_values = [-4.05, -2.00, 0.05, 1.98, 4.18, 6.20, 8.22, 10.18, 11.08, 12.25, 13.10, 14.28, 15.20, 16.18, 16.9, 17.35]
y_values = [0.00700, 0.00650, 0.00650, 0.00680, 0.00760, 0.00680, 0.00800, 0.01050, 0.01140, 0.01250, 0.01300, 0.01620, 0.01870, 0.02180, 0.02440, 0.02750]

# Plotting
plt.figure(figsize=(10, 6))

plt.scatter(x_values, y_values, marker='D', s=20, label=r'$C_D$ - Numerical', color='black', facecolor='none')

plt.errorbar(x, y, yerr=ci_y, xerr=ci_x, fmt='s', ecolor='red', elinewidth=2, capsize=5, capthick=2, markersize=4, markerfacecolor='none', mec='red', mew=1, alpha=1)
plt.plot(x, y, '--', color='red', alpha=0.5)
plt.scatter(x, y, marker='s', color='red', facecolor='none', s=20, label=r'$C_D$ - Experimental')

plt.xlabel(r'Angle of Attack $\alpha$ [°]')
plt.ylabel(r'$C_D$')
plt.ylim(-0.15, 0.20)
plt.xlim(-5, 20)
plt.legend()

plt.savefig('../results/CD_confidence_intervals.png', dpi=600)

plt.show()