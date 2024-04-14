import pandas as pd
import statistics
import glob
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

base_folder_paths = [
    '../data/mcs_results/re_5591000/',
    '../data/mcs_results/re_5970000/',
    '../data/mcs_results/re_6725000/'
]

for base_folder_path in base_folder_paths:
    re_value = os.path.basename(base_folder_path.strip('/')).split('_')[1]

    output_png_folder = f'../results/PDF/CL/{re_value}/'

    os.makedirs(output_png_folder, exist_ok=True)

    aoa_folders = glob.glob(f'{base_folder_path}aoa_*')

    aoa_folders = [(float(os.path.basename(folder).split('_')[1].replace('p', '.')), folder) for folder in aoa_folders]

    aoa_folders.sort(key=lambda x: x[0])

    results = []

    for aoa, folder_path in aoa_folders:
        files = glob.glob(f'{folder_path}/*.dat')
        values = []

        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    match = re.search(r'CL\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                    if match:
                        value = float(match.group(1))
                        values.append(value)

        values = np.array(values)
        values = values[np.isfinite(values)]

        if len(values) > 0:
            std_dev = np.std(values)
            variance = np.var(values)
            mean = np.mean(values)
            median = np.median(values)

            plt.figure(figsize=(6, 5))
            sns.kdeplot(values, fill=True, alpha=0.5, linewidth=2)
            plt.axvline(mean, color='red', linestyle='-', label=r'$\mu$')
            plt.axvline(mean - std_dev, color='orange', linestyle='--', label=r'1$\sigma$')
            plt.axvline(mean + std_dev, color='orange', linestyle='--')
            plt.axvline(mean - 2 * std_dev, color='green', linestyle=':', label=r'2$\sigma$')
            plt.axvline(mean + 2 * std_dev, color='green', linestyle=':')
            
            plt.title(f'PDF for AOA {aoa:.2f}, Mean ($\mu$) = {mean:.2f} & Std Dev ($\sigma$) = {std_dev:.2f}')
            plt.xlabel('CL Value')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(f'{output_png_folder}{re_value}_AOA_{aoa:.2f}_PDF.png')
            plt.close()

            results.append([aoa, std_dev, variance, mean, median])

    df_results = pd.DataFrame(results, columns=['AOA', 'Std Dev', 'Variance', 'Mean', 'Median'])

    df_results = df_results.sort_values(by='AOA')

    os.makedirs('../data/CL/', exist_ok=True)
    output_file_name = f'../data/CL/distribution_re_{re_value}.txt'

    df_results.to_csv(output_file_name, sep='\t', index=False)

    print(f'Results saved to {output_file_name}')
