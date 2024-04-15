import pandas as pd
import numpy as np
import glob
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

base_folder_paths = [
    '../data/mcs_results/re_5591000/',
    '../data/mcs_results/re_6725000/'
]

aoa_data = {}

for base_folder_path in base_folder_paths:
    normalized_path = os.path.normpath(base_folder_path)
    
    path_components = normalized_path.split(os.sep)
    
    re_index = path_components.index('mcs_results') + 1
    re_folder = path_components[re_index]
    
    re_value = os.path.basename(base_folder_path.strip('/')).split('_')[1]

    aoa_folders = glob.glob(f'{base_folder_path}aoa_*')

    for folder_path in aoa_folders:
        aoa = os.path.basename(folder_path).split('_')[1].replace('p', '.')

        if aoa not in aoa_data:
            aoa_data[aoa] = {}
        aoa_data[aoa][re_value] = []

        files = glob.glob(f'{folder_path}/*.dat')

        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    match = re.search(r'CL\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                    if match:
                        value = float(match.group(1))
                        aoa_data[aoa][re_value].append(value)

output_fig_folder = '../results/CDF/CL/'
os.makedirs(output_fig_folder, exist_ok=True)

for aoa, re_values in aoa_data.items():
    plt.figure(figsize=(8, 6))
    
    for re_value, values in re_values.items():
        values = np.array(values)
        values = values[np.isfinite(values)]

        if len(values) > 0:
            re_value_float = float(re_value)
            re_value_sci = f'{re_value_float:.3e}'
            base, exponent = re_value_sci.split('e+')
            sns.ecdfplot(values, label = f'$Re = {base} \\times 10^{{{int(exponent)}}}$')

    plt.title(rf'CDF of $C_D$ at $\alpha$ = {aoa}°')
    plt.xlabel(r'$C_L$')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_fig_folder}Overlay_CDF_AOA_{aoa}.png')
    plt.close()

print("CDF plots created for each AoA across different REs.")
