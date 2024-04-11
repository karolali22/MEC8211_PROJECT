import pandas as pd
import statistics
import glob
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Base directory containing all the "aoa_" folders
base_folder_paths = [
    'mcs_results/re_5793000/',
    'mcs_results/re_5970000/',
    'mcs_results/re_6154000/'
]

for base_folder_path in base_folder_paths:
    # Extract the "re_" value from the base_folder_path
    re_value = base_folder_path.split('/')[1]

    # Define the output folder for the PNG files
    output_png_folder = f'output_pngs/{re_value}/'
    # Create the output directory if it doesn't exist
    os.makedirs(output_png_folder, exist_ok=True)

    # Find all directories with "aoa_" in their name
    aoa_folders = glob.glob(f'{base_folder_path}aoa_*')

    # Prepare a list to store results
    results = []

    for folder_path in aoa_folders:
        files = glob.glob(f'{folder_path}/*.dat')
        values = []

        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    match = re.search(r'CD\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                    if match:
                        value = float(match.group(1))
                        values.append(value)

        # Clean data: Remove NaNs and infinite values
        values = np.array(values)
        values = values[np.isfinite(values)]

        if len(values) > 0:
            std_dev = np.std(values)
            variance = np.var(values)
            mean = np.mean(values)
            median = np.median(values)

            # Plotting
            plt.figure(figsize=(6, 5))
            sns.kdeplot(values, fill=True, alpha=0.5, linewidth=2)
            plt.axvline(mean, color='red', linestyle='-', label=r'$\mu$')
            plt.axvline(mean - std_dev, color='orange', linestyle='--', label=r'1$\sigma$')
            plt.axvline(mean + std_dev, color='orange', linestyle='--')
            plt.axvline(mean - 2 * std_dev, color='green', linestyle=':', label=r'2$\sigma$')
            plt.axvline(mean + 2 * std_dev, color='green', linestyle=':')
            
            # Extract the aoa value from the folder name and replace 'p' with '.'
            aoa = os.path.basename(folder_path).split('_')[1].replace('p', '.')

            plt.title(f'PDF for AOA {aoa}, Mean ($\mu$) = {mean:.2f} & Std Dev ($\sigma$) = {std_dev:.2f}')
            plt.xlabel('CD Value')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(f'{output_png_folder}{re_value}_AOA_{aoa}_PDF.png')
            plt.close()

            # Append the results
            results.append([aoa, std_dev, variance, mean, median])

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results, columns=['AOA', 'Std Dev', 'Variance', 'Mean', 'Median'])

    # Define the output file name using the "re_" value
    output_file_name = f'{output_png_folder}distribution_{re_value}.txt'

    # Save the DataFrame to a .txt file
    df_results.to_csv(output_file_name, sep='\t', index=False)

    print(f'Results saved to {output_file_name}')
