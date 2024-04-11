import os
import shutil
import numpy as np

NUM_SIMULATIONS = 50
REYNOLDS_NUMBERS = [5.793e6, 5.970e6, 6.154e6]

aoa_means = [-4.05, -2.00, 0.05, 1.98, 4.18, 6.20, 8.22, 10.18, 11.08, 12.25, 13.10, 14.28, 15.20, 16.18, 16.9, 17.35, 17.65, 18.65]
mach_mean = 0.15
aoa_std = 0.01
mach_std = 0.0025

mach_values = np.random.normal(mach_mean, mach_std, NUM_SIMULATIONS)

steady_input_path = 'steady_inputs'
unsteady_input_path = 'unsteady_inputs'
output_folder = 'FLOW_OUTPUT'
results_folder = 'mcs_results'

os.makedirs(results_folder, exist_ok=True)

def update_input_file(file_path, aoa, mach, reynolds):
    updated_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("AOA="):
                updated_line = f"AOA={aoa}\n"
                updated_lines.append(updated_line)
            elif line.startswith("MACH="):
                updated_line = f"MACH={mach}\n"
                updated_lines.append(updated_line)
            elif line.startswith("REYNOLDS="):
                updated_line = f"REYNOLDS={int(reynolds)}\n"
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

for reynolds in REYNOLDS_NUMBERS:
    reynolds_folder = f're_{reynolds:.0f}'
    reynolds_path = os.path.join(results_folder, reynolds_folder)
    os.makedirs(reynolds_path, exist_ok=True)

    for mean_aoa in aoa_means:
        aoa_values = np.random.normal(mean_aoa, aoa_std, NUM_SIMULATIONS)
        
        subfolder_name = f'aoa_{mean_aoa:.2f}'
        subfolder_path = os.path.join(reynolds_path, subfolder_name.replace('.', 'p'))
        os.makedirs(subfolder_path, exist_ok=True)

        for aoa, mach in zip(aoa_values, mach_values):
            update_input_file(steady_input_path, aoa, mach, reynolds)
            update_input_file(unsteady_input_path, aoa, mach, reynolds)

            os.system('~/champs-development/bin/champs__flow -nl 1 -f steady_inputs')
            os.system('~/champs-development/bin/champs__flow -nl 1 -f unsteady_inputs')

            original_file = os.path.join(output_folder, 'forces_unsteady.dat')
            new_file_name = f'forces_unsteady_aoa_{aoa:.2f}_mach_{mach:.4f}.dat'
            new_file = os.path.join(subfolder_path, new_file_name)

            shutil.copy(original_file, new_file)

print("Monte Carlo simulations completed with predefined normal distributions, organized by AoA mean values and Reynolds numbers.")
