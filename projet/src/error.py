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
    cl_theoretical_values = [2 * np.pi * np.sin(np.radians(aoa)) for aoa in aoa_values]

    errors = [abs(cl - cl_th) for cl, cl_th in zip(cl_values, cl_theoretical_values)]
    l1_norm = np.sum(errors)
    l2_norm = np.sqrt(np.sum([e**2 for e in errors]))
    linf_norm = max(errors)

    return l1_norm, l2_norm, linf_norm

def plot_cl_values(aoa_values, cl_values, cl_theoretical_values, folder_name):
    plt.figure(figsize=(10, 5))
    plt.scatter(aoa_values, cl_values, color='blue', label='Simulation CL')
    plt.scatter(aoa_values, cl_theoretical_values, color='red', label='Theoretical CL', marker='x')
    plt.title(f'CL Values for Folder: {folder_name}')
    plt.xlabel('Angle of Attack (Degrees)')
    plt.ylabel('CL Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_convergence_order(errors):
    p = np.log( (abs(errors[0] - errors[1])) /( abs(errors[1] - errors[2])))/np.log(2)
    return p

folder_paths = ['../data/convergence_results/65x65', '../data/convergence_results/129x129', '../data/convergence_results/257x257']
errors_dict = {}

for folder in folder_paths:
    l1_norm, l2_norm, linf_norm = calculate_errors(folder)
    errors_dict[folder] = {'L1': l1_norm, 'L2': l2_norm, 'Linf': linf_norm}

l1_errors = [errors_dict[folder]['L1'] for folder in folder_paths]
l2_errors = [errors_dict[folder]['L2'] for folder in folder_paths]
linf_errors = [errors_dict[folder]['Linf'] for folder in folder_paths]

p_l1 = calculate_convergence_order(l1_errors)
p_l2 = calculate_convergence_order(l2_errors)
p_linf = calculate_convergence_order(linf_errors)

print(f"Order of convergence for L1 norm (p): {p_l1:.2f}")
print(f"Order of convergence for L2 norm (p): {p_l2:.2f}")
print(f"Order of convergence for Linf norm (p): {p_linf:.2f}")
