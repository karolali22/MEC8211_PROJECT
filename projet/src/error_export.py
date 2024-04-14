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
    cl_theoretical_values = [2 * np.pi * np.radians(aoa) for aoa in aoa_values]

    errors = [abs(cl - cl_th) for cl, cl_th in zip(cl_values, cl_theoretical_values)]
    l1_norm = np.sum(errors)
    l2_norm = np.sqrt(np.sum([e**2 for e in errors]))
    linf_norm = max(errors)

    return l1_norm, l2_norm, linf_norm, aoa_values, cl_values, cl_theoretical_values

def save_errors_to_file(errors_dict, filename="error_summary.txt"):
    with open(filename, "w") as file:
        file.write("Mesh Size\tL1 Norm\tL2 Norm\tLinf Norm\n")
        for key, values in errors_dict.items():
            file.write(f"{key}\t{values['L1']:.6f}\t{values['L2']:.6f}\t{values['Linf']:.6f}\n")

folder_paths = ['../data/convergence_results/65x65', '../data/convergence_results/129x129', '../data/convergence_results/257x257']
errors_dict = {}

for folder in folder_paths:
    l1_norm, l2_norm, linf_norm, _, _, _ = calculate_errors(folder)
    errors_dict[folder] = {'L1': l1_norm, 'L2': l2_norm, 'Linf': linf_norm}

save_errors_to_file(errors_dict)

print("Error data has been saved to 'error_summary.txt'.")
