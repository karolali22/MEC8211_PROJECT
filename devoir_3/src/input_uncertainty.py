"""

MEC8211 - Homework 3 - Group 4

"""

import pandas as pd
import statistics

files = ['../data/simulation_results_11.txt',
         '../data/simulation_results_28.txt',
         '../data/simulation_results_82.txt',
         '../data/simulation_results_1128.txt',
         '../data/simulation_results_8211.txt']

std_devs = []
variances = []
means = []

for file in files:
    df = pd.read_csv(file, sep='\s+', header=None)

    if df.shape[1] < 3:
        print(f"File {file} does not have enough columns.")
        continue

    third_column = df.iloc[:, 2]

    std_dev = third_column.std()
    variance = third_column.var()
    mean = third_column.mean()
    
    std_devs.append(std_dev)
    variances.append(variance)
    means.append(mean)

    print(f"File: {file}, Standard Deviation of permeability: {std_dev}, Variance of permeability: {variance}, Mean permeability: {mean}")

mean_std_dev = sum(std_devs) / len(std_devs)
mean_var = sum(variances) / len(variances)
mean_mean = sum(means) / len(means)
median_seeds = statistics.median(means)

print(f"\nMean Standard Deviation across all seeds: {mean_std_dev}")
print(f"\nMean Variance across all seeds: {mean_var}")
print(f"\nMean Mean across all seeds: {mean_mean}")
print(f"\nMedian across all seeds: {median_seeds}")
