"""

MEC8211 - Homework 3 - Group 4

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re 

files = [
    '../data/simulation_results_11.txt',
    '../data/simulation_results_28.txt',
    '../data/simulation_results_82.txt',
    '../data/simulation_results_1128.txt',
    '../data/simulation_results_8211.txt'
]

third_columns = []
file_identifiers = []

pattern = re.compile(r'simulation_results_(\d+)\.txt')

for file in files:
    match = pattern.search(file)
    if match:
        file_identifiers.append(match.group(1))
    else:
        file_identifiers.append("Unknown")
    
    df = pd.read_csv(file, sep=r'\s+', header=None)
    
    if df.shape[1] < 3:
        print(f"File {file} does not have enough columns.")
        continue

    third_columns.append(df.iloc[:, 2])

for i, column in enumerate(third_columns):
    mean = column.mean()
    std_dev = column.std()
    plt.figure(figsize=(6, 5))
    sns.kdeplot(column, fill=True, alpha=0.5, linewidth=2)
    plt.axvline(mean, color='red', linestyle='-', label=r'$\mu$')
    plt.axvline(mean - std_dev, color='orange', linestyle='--', label=r'1$\sigma$')
    plt.axvline(mean + std_dev, color='orange', linestyle='--')
    plt.axvline(mean - 2 * std_dev, color='green', linestyle=':', label=r'2$\sigma$')
    plt.axvline(mean + 2 * std_dev, color='green', linestyle=':')
    
    plt.title(f'PDF for Seed {file_identifiers[i]}, Mean ($\mu$) = {mean:.2f} & Std Dev ($\sigma$) = {std_dev:.2f}')
    plt.xlabel(r'Permeability k [$\mu$m$^2$]')
    plt.ylabel('Density')
    plt.legend()
    #plt.show()

# Concatenate all third column data into a single DataFrame or Series
combined_data = pd.concat(third_columns, ignore_index=True)

# Plot the combined PDF
plt.figure(figsize=(6, 5))
sns.kdeplot(combined_data, fill=True, alpha=0.5, linewidth=2)
plt.xlabel(r'Permeability k [$\mu$m$^2$]')
plt.ylabel('Density')

# Calculate and plot mean and standard deviation lines for the combined data
combined_mean = combined_data.mean()
print(combined_mean)
combined_std_dev = combined_data.std()
print(combined_std_dev)
plt.title(f'Combined Seed PDF, Mean ($\mu$) = {combined_mean:.2f} & Std Dev ($\sigma$) = {combined_std_dev:.2f}')
plt.axvline(combined_mean, color='red', linestyle='-', label='$\mu$')
plt.axvline(combined_mean - combined_std_dev, color='orange', linestyle='--', label='1$\sigma$')
plt.axvline(combined_mean + combined_std_dev, color='orange', linestyle='--')
plt.axvline(combined_mean - 2 * combined_std_dev, color='green', linestyle=':', label='2$\sigma$')
plt.axvline(combined_mean + 2 * combined_std_dev, color='green', linestyle=':')

plt.legend()
plt.show()
