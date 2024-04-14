import os
import numpy as np
import re
import matplotlib.pyplot as plt

def extract_values_from_line(line):
    match = re.search(r'(\d+\.\d+e[+-]\d+)', line)
    if match:
        return float(match.group(1))
    else:
        return None

def calculate_errors(folder_path):
    temp_aoa = {}
    temp_cl = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.dat'):
            current_aoa = None
            with open(os.path.join(folder_path, filename), 'r') as file:
                for line in file:
                    if 'ALPHA =' in line:
                        current_aoa = extract_values_from_line(line)
                    elif 'CL =' in line and current_aoa is not None:
                        cl = extract_values_from_line(line)
                        if cl is not None:
                            temp_cl[current_aoa] = cl
                            temp_aoa[current_aoa] = current_aoa

    aoa_values = sorted(temp_aoa.keys())
    cl_values = [temp_cl[aoa] for aoa in aoa_values]
    #cl_theoretical_values = [2 * np.pi * np.sin(np.radians(aoa)) for aoa in aoa_values]
    cl_theoretical_values = [2 * np.pi * np.radians(aoa) for aoa in aoa_values]

    errors = [abs(cl - cl_th) for cl, cl_th in zip(cl_values, cl_theoretical_values)]
    l1_norm = np.sum(errors)
    l2_norm = np.sqrt(np.sum([e**2 for e in errors]))
    linf_norm = max(errors)

    return aoa_values, cl_values, cl_theoretical_values, l1_norm, l2_norm, linf_norm

def plot_all_cl_values(data_dict):
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^']
    colors = ['blue', 'green', 'red']
    
    for i, (key, (aoa_values, cl_values, cl_theoretical_values)) in enumerate(data_dict.items()):
        plt.scatter(aoa_values, cl_values, color=colors[i], marker=markers[i], label=f'Simulation {key} CL')

    plt.plot(aoa_values, cl_theoretical_values, color='black', linestyle='--', label=f'Theoretical CL')
    plt.title('CL Values Across Different Mesh Resolutions')
    plt.xlabel('Angle of Attack (Degrees)')
    plt.ylabel('CL Value')
    plt.legend()
    plt.grid(True)
    plt.show()

folder_paths = ['../data/convergence_results/65x65', '../data/convergence_results/129x129', '../data/convergence_results/257x257']
cl_data_dict = {}

for folder in folder_paths:
    aoa_values, cl_values, cl_theoretical_values, _, _, _ = calculate_errors(folder)
    cl_data_dict[folder] = (aoa_values, cl_values, cl_theoretical_values)

plot_all_cl_values(cl_data_dict)
