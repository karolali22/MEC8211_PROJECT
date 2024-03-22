"""

MEC8211 - Homework 3 - Group 4

"""

import numpy as np
import subprocess
import os

mean = 0.9
std_dev = 7.5e-3
num_samples = 100
poro_samples = np.random.normal(mean, std_dev, num_samples)

main_matlab_script_path = os.path.abspath('launch_simulationLBM.m')

temp_script_dir = './temp_matlab_scripts'
os.makedirs(temp_script_dir, exist_ok=True)

for i, poro_value in enumerate(poro_samples, start=1):
    temp_script_name = f'temp_script_{i}.m'
    temp_script_path = os.path.join(temp_script_dir, temp_script_name).replace('\\', '/')

    with open(temp_script_path, 'w') as file:
        file.write(f'poro={poro_value};\n')
        with open(main_matlab_script_path, 'r') as main_script:
            file.write(main_script.read())
    
    matlab_command = f"matlab -batch \"run('{temp_script_path}');\""
    result = subprocess.run(matlab_command, shell=True)

    if result.returncode != 0:
        print(f"Error: MATLAB command failed for sample {i}. Check the MATLAB script and environment.")

    os.remove(temp_script_path)
